"""Tests for ``eval/curve.py``: the retry-budget curve, and the calls each cap pays for.

Two properties are under test. **The curve partitions the loop's cases at every cap**: a case
that entered the loop has either passed by that cap or hit it, so ``passed + hit_cap ==
entered``, and a case that never entered (``not_case``, ``low_score``) or errored is left out.
**``calls_at_cap`` charges a case only for the drafts it would have run**: the grading, plus one
extract and one review per draft up to the cap.

No corpus, no API. Records here are small hand-built dicts carrying only the keys ``curve.py``
reads, built by the helpers below from a case's concern counts -- the numbers on ``--show``'s
arrow line.

**Note on authorship.** You wrote ``first_clean_draft`` and ``retry_budget_curve``; the agent
wrote ``calls_at_cap`` and these helpers. Every ``assert`` and ``pytest.raises`` is yours.
"""

import pytest  # noqa: F401

from curator.eval.curve import calls_at_cap, first_clean_draft, retry_budget_curve  # noqa: F401
from curator.schemas import Role
from curator.state import TerminalState

#: The first installment's cap, and the default for every record built here.
MAX_REFINES = 4

#: The first installment's ten cases as concern counts per draft, ``--show``'s arrow lines.
#: Counts only, no case text, so safe to commit.
FIRST_TEN = [
    [8, 2, 4, 2, 5],
    [3, 1, 0],
    [7, 3, 3, 4, 3],
    [3, 4, 3, 5, 4],
    [2, 1, 0],
    [3, 1, 2, 2, 2],
    [3, 3, 1, 2, 3],
    [2, 0],
    [2, 1, 3, 0],
    [2, 1, 2, 2, 1],
]


# --------------------------------------------------------------------------------------
# Building records
# --------------------------------------------------------------------------------------


def _call(role: Role) -> dict:
    """One entry of a record's ``usage``, as read back from JSON. The token counts are stand-ins."""
    return {
        "role": role.value,
        "model": "stub",
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "output_tokens": 10,
        "reasoning_tokens": 0,
    }


def case(concerns: list[int], source: str = "case-1", max_refines: int = MAX_REFINES) -> dict:
    """A record of a case that entered the loop, one draft per entry of ``concerns``.

    ``case([2, 1, 3, 0])`` is case 9: four drafts, the editor raising 2, 1, 3 and 0 concerns.
    Its ``usage`` is the grading call, then an extract and a review per draft, in the order the
    graph makes them. Drafts and concerns are stand-ins; ``curve.py`` only counts them.

    The graph stops a case at its first clean draft or at the cap, and nowhere else. Concern
    counts that could not come from a run -- a 0 before the end, more drafts than the cap allows,
    a case still flagged before reaching the cap -- raise ``ValueError``, so a typo in a test
    cannot build a record no run would produce.
    """
    if not concerns:
        raise ValueError("a case that entered the loop has at least one draft; see not_case()")
    if 0 in concerns[:-1]:
        raise ValueError(f"{concerns}: the loop stops at the first 0")
    if len(concerns) > max_refines + 1:
        raise ValueError(f"{concerns}: more drafts than max_refines={max_refines} allows")
    passed = concerns[-1] == 0
    if not passed and len(concerns) < max_refines + 1:
        raise ValueError(f"{concerns}: still flagged, so it would have gone round again")

    return {
        "source": source,
        "terminal_state": (TerminalState.PASSED if passed else TerminalState.HIT_MAX_REFINES).value,
        "config": {"max_refines": max_refines},
        "drafts": [{} for _ in concerns],
        "verdicts": [
            {"concerns": [{"section": "stub", "explanation": "stub"}] * count}
            for count in concerns
        ],
        "usage": [_call(Role.GRADER)]
        + [_call(role) for _ in concerns for role in (Role.EXTRACTOR, Role.EDITOR)],
    }


def not_case(source: str = "case-1", max_refines: int = MAX_REFINES) -> dict:
    """A record of a case that stopped at grading: no drafts, no verdicts, one grading call.

    A ``low_score`` record has the same shape; only ``terminal_state`` differs.
    """
    return {
        "source": source,
        "terminal_state": TerminalState.NOT_CASE.value,
        "config": {"max_refines": max_refines},
        "drafts": [],
        "verdicts": [],
        "usage": [_call(Role.GRADER)],
    }


def errored(source: str = "case-1", max_refines: int = MAX_REFINES) -> dict:
    """A record of a case whose run raised, as ``build_error_record`` writes it.

    It has ``error`` and ``config``, but no ``terminal_state``, ``drafts``, ``verdicts`` or
    ``usage``.
    """
    return {
        "source": source,
        "error": {"type": "RefusalError", "message": "stub"},
        "config": {"max_refines": max_refines},
    }


def first_ten() -> list[dict]:
    """The first installment rebuilt from ``FIRST_TEN``, as ``case-1`` to ``case-10``."""
    return [case(concerns, f"case-{n}") for n, concerns in enumerate(FIRST_TEN, start=1)]


# --------------------------------------------------------------------------------------
# Your tests  [yours] -- what D33 asks to cover:
#   first_clean_draft   case 9's shape and case 1's shape: the column you worked out by hand
#   retry_budget_curve  your hand table over first_ten(); passed + hit_cap == entered at every
#                       cap; adding not_case() and errored() records changes nothing
#   calls_at_cap        the calls per cap for case 9's shape; a cap above max_refines raises
#                       ValueError; so does a record with one draft removed, e.g.
#                       dict(record, drafts=record["drafts"][:-1])
# --------------------------------------------------------------------------------------

def test_first_clean_draft() -> None:
    """The first draft with no concerns, or None if none did."""
    assert first_clean_draft(case([8, 2, 4, 2, 5])) is None
    assert first_clean_draft(case([3, 1, 0])) == 2
    assert first_clean_draft(case([7, 3, 3, 4, 3])) is None
    assert first_clean_draft(case([3, 4, 3, 5, 4])) is None
    assert first_clean_draft(case([2, 1, 0])) == 2
    assert first_clean_draft(case([3, 1, 2, 2, 2])) is None
    assert first_clean_draft(case([3, 3, 1, 2, 3])) is None
    assert first_clean_draft(case([2, 0])) == 1
    assert first_clean_draft(case([2, 1, 3, 0])) == 3
    assert first_clean_draft(case([2, 1, 2, 2, 1])) is None


def test_retry_budget_curve() -> None:
    """The retry-budget curve for the first installment, and its invariants."""
    records = first_ten()
    curve = retry_budget_curve(records)
    assert curve == {
        0: {"entered": 10, "passed": 0, "hit_cap": 10},
        1: {"entered": 10, "passed": 1, "hit_cap": 9},
        2: {"entered": 10, "passed": 3, "hit_cap": 7},
        3: {"entered": 10, "passed": 4, "hit_cap": 6},
        4: {"entered": 10, "passed": 4, "hit_cap": 6},
    }
    for cap in range(MAX_REFINES + 1):
        assert curve[cap]["passed"] + curve[cap]["hit_cap"] == curve[cap]["entered"]

    # Adding not_case() and errored() records changes nothing
    records += [not_case("case-11"), errored("case-12")]
    assert retry_budget_curve(records) == curve


def test_calls_at_cap() -> None:
    """The calls per cap for case 9's shape, and the errors that arise from bad input."""
    record = case([2, 1, 3, 0])

    assert [len(calls_at_cap(record, cap)) for cap in range(MAX_REFINES + 1)] == [3, 5, 7, 9, 9]

    with pytest.raises(ValueError):
        calls_at_cap(record, MAX_REFINES + 1)
    with pytest.raises(ValueError):
        calls_at_cap(dict(record, drafts=record["drafts"][:-1]), MAX_REFINES)
