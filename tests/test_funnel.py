"""Tests for ``funnel.py``: every case attempted lands in exactly one bucket.

The property under test is a **partition**. Every record is counted once, in one bucket, and the
counts add up to the number of records. That is the property the notebook broke: ``dropna()``
made fifteen cases vanish from every count without appearing in any (PLAN 2.7).

No corpus, no API. Records here are small hand-built dicts carrying only the keys the funnel
reads, built by the helpers below. The last test writes real record files into a temporary
folder, so it also checks that ``load_batch`` and ``funnel`` agree with each other.

**Note on authorship.** You wrote ``funnel``; the agent wrote ``load_batch`` and these helpers.
Every ``assert`` and ``pytest.raises`` is yours.
"""

from pathlib import Path

import pytest

from curator.funnel import funnel  # noqa: F401
from curator.records import load_batch, write_batch_record  # noqa: F401
from curator.state import TerminalState  # noqa: F401

#: The five bucket names, as the funnel returns them.
BUCKETS = [state.value for state in TerminalState] + ["error"]


# --------------------------------------------------------------------------------------
# Building records
# --------------------------------------------------------------------------------------


def finished(state: TerminalState, source: str = "case-1") -> dict:
    """A record of a case that left the graph through ``state``.

    ``finished(TerminalState.PASSED)`` gives ``{"source": "case-1", "terminal_state": "passed",
    ...}`` -- the string, as a record read back from JSON holds it, not the enum member.
    """
    return {
        "source": source,
        "terminal_state": state.value,
        "started_at": "2026-10-24T09:00:00+00:00",
    }


def errored(source: str = "case-1", started_at: str = "2026-10-24T09:00:00+00:00") -> dict:
    """A record of a case whose run raised: an ``error``, and no ``terminal_state``."""
    return {
        "source": source,
        "error": {"type": "RefusalError", "message": "stub"},
        "started_at": started_at,
    }


def mixed_batch() -> list[dict]:
    """Seven records, at least one in every bucket, two in some."""
    return [
        finished(TerminalState.PASSED, "case-1"),
        finished(TerminalState.PASSED, "case-2"),
        finished(TerminalState.LOW_SCORE, "case-3"),
        finished(TerminalState.NOT_CASE, "case-4"),
        finished(TerminalState.HIT_MAX_REFINES, "case-5"),
        finished(TerminalState.HIT_MAX_REFINES, "case-6"),
        errored("case-7"),
    ]


def write_batch(batch_dir: Path, records: list[dict]) -> None:
    """Write ``records`` into ``batch_dir`` as real files, the way a batch run does."""
    for record in records:
        write_batch_record(record, batch_dir)


# --------------------------------------------------------------------------------------
# The partition  [yours]
# --------------------------------------------------------------------------------------


def test_every_bucket_is_present_even_when_empty():
    """An empty batch still reports all five buckets, each at zero.

    Available: ``BUCKETS`` lists the five names.
    """
    assert funnel([]) == {bucket: 0 for bucket in BUCKETS}


def test_one_record_lands_in_exactly_one_bucket():
    """A single record is counted once: its bucket is 1 and every other bucket is 0.

    Worth checking for more than one kind of record -- at least one finished and one errored.
    """
    finished_record = finished(TerminalState.PASSED, "case-1")
    errored_record = errored("case-2")
    assert funnel([finished_record]) == {
        bucket: 1 if bucket == TerminalState.PASSED.value else 0 for bucket in BUCKETS
    }
    assert funnel([errored_record]) == {bucket: 1 if bucket == "error" else 0 for bucket in BUCKETS}


def test_the_counts_add_up_to_the_number_of_records():
    """The partition property itself, over ``mixed_batch()``. The ``dropna()`` claim."""
    records = mixed_batch()
    counts = funnel(records)
    assert sum(counts.values()) == len(records)

def test_an_unknown_terminal_state_fails_loudly():
    """Your decision: a stop reason that is not one of the four raises, rather than becoming a
    sixth bucket. ``finished`` only takes real members, so build this record by hand.
    """
    bad_record = {
        "source": "case-1",
        "terminal_state": "unknown",
        "started_at": "2026-10-24T09:00:00+00:00",
    }
    with pytest.raises(KeyError):
        funnel([bad_record])

# --------------------------------------------------------------------------------------
# Through real files  [yours] -- tmp_path is a pytest fixture: a fresh empty folder per test
# --------------------------------------------------------------------------------------


def test_a_retried_case_is_counted_once(tmp_path):
    """A case that errored, was retried on resume and then finished, is one case, not two.

    Setup idea: ``write_batch(tmp_path, [...])`` with an ``errored("case-1")`` and a
    ``finished(..., "case-1")`` for the same source, plus anything else you like. Then
    ``funnel(load_batch(tmp_path))``. Claim how that case is counted, and the total.
    """
    write_batch(tmp_path, [errored("case-1"), finished(TerminalState.PASSED, "case-1")])
    counts = funnel(load_batch(tmp_path))
    assert counts["error"] == 0
    assert counts[TerminalState.PASSED.value] == 1
    assert sum(counts.values()) == 1
