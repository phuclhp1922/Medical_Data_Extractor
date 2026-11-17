"""Tests for ``eval/convergence.py``: is the editor converging, or moving the goalposts?

Two counts over consecutive reviews of one case (Phase 2 step 2, D34):

- ``concern_trend(records) -> dict[str, int]``: how many review-to-review steps had fewer, the
  same, or more concerns, as ``{"down": n, "flat": n, "up": n}``.
- ``section_repeats(records) -> dict[str, dict[str, int]]``: for each of the five sections, how
  many concerns had that section already flagged in the review before (``"repeated"``) and how
  many did not (``"new"``). Every section is present, zeros included, as the funnel reports
  every bucket.

Only cases that entered the loop count: an error record or a case stopped at grading has no
consecutive reviews. A first review has nothing before it, so its concerns are neither repeated
nor new.

No corpus, no API. Records here are small hand-built dicts carrying only the keys these counts
read, built by the helpers below from the sections flagged in each review.

**Note on authorship.** The agent wrote these helpers. The functions in ``convergence.py``, and
every ``assert`` here, are yours.
"""

from curator.eval.convergence import concern_trend, section_repeats
from curator.schemas import ChecklistSection
from curator.state import TerminalState

CPQ = ChecklistSection.CASE_PRESENTATION_QUALITY.value
DR = ChecklistSection.DIAGNOSTIC_REASONING.value
FD = ChecklistSection.FINAL_DIAGNOSIS.value
HAL = ChecklistSection.HALLUCINATIONS.value
SF = ChecklistSection.SOURCE_FIDELITY.value

#: The five section names, as the counts report them.
SECTIONS = [section.value for section in ChecklistSection]

#: The first installment's cap, and the default for every record built here.
MAX_REFINES = 4

#: The first installment's ten cases: the sections flagged in each review, one per concern.
#: Labels only, no explanation text, so safe to commit.
FIRST_TEN = [
    [[CPQ, CPQ, DR, DR, DR, DR, FD, HAL], [CPQ, DR], [FD, DR, DR, SF], [CPQ, DR],
     [SF, CPQ, DR, FD, HAL]],
    [[DR, DR, DR], [DR], []],
    [[SF, DR, DR, DR, SF, FD, HAL], [CPQ, DR, FD], [CPQ, CPQ, DR], [DR, HAL, DR, FD],
     [CPQ, DR, FD]],
    [[DR, DR, SF], [SF, CPQ, DR, FD], [CPQ, DR, FD], [CPQ, CPQ, DR, DR, SF], [DR, FD, SF, HAL]],
    [[HAL, SF], [DR], []],
    [[DR, SF, HAL], [DR], [SF, DR], [DR, DR], [SF, DR]],
    [[CPQ, DR, DR], [SF, CPQ, DR], [DR], [DR, DR], [DR, SF, DR]],
    [[DR, SF], []],
    [[DR, DR], [DR], [DR, DR, SF], []],
    [[DR, CPQ], [DR], [DR, DR], [DR, DR], [DR]],
]


# --------------------------------------------------------------------------------------
# Building records
# --------------------------------------------------------------------------------------


def case(rounds: list[list[str]], source: str = "case-1", max_refines: int = MAX_REFINES) -> dict:
    """A record of a case that entered the loop, one review per entry of ``rounds``.

    Each entry lists the sections the editor flagged in that review, one per concern.
    ``case([[DR, SF], [DR], []])`` is three reviews raising 2, 1 and 0 concerns, so it passed at
    draft 2. Drafts and explanations are stand-ins; nothing here reads them.

    As with ``test_curve.case``, rounds no run could produce raise ``ValueError``: a clean review
    before the last, more reviews than the cap allows, a case still flagged before reaching the
    cap, or a section name that is not one of the five.
    """
    if not rounds:
        raise ValueError("a case that entered the loop has at least one review; see not_case()")
    if [] in rounds[:-1]:
        raise ValueError(f"{rounds}: the loop stops at the first clean review")
    if len(rounds) > max_refines + 1:
        raise ValueError(f"{rounds}: more reviews than max_refines={max_refines} allows")
    passed = rounds[-1] == []
    if not passed and len(rounds) < max_refines + 1:
        raise ValueError(f"{rounds}: still flagged, so it would have gone round again")
    unknown = {section for sections in rounds for section in sections} - set(SECTIONS)
    if unknown:
        raise ValueError(f"{sorted(unknown)}: not a checklist section")

    return {
        "source": source,
        "terminal_state": (TerminalState.PASSED if passed else TerminalState.HIT_MAX_REFINES).value,
        "config": {"max_refines": max_refines},
        "drafts": [{} for _ in rounds],
        "verdicts": [
            {"concerns": [{"section": section, "explanation": "stub"} for section in sections]}
            for sections in rounds
        ],
    }


def not_case(source: str = "case-1", max_refines: int = MAX_REFINES) -> dict:
    """A record of a case that stopped at grading: no drafts and no reviews.

    A ``low_score`` record has the same shape; only ``terminal_state`` differs.
    """
    return {
        "source": source,
        "terminal_state": TerminalState.NOT_CASE.value,
        "config": {"max_refines": max_refines},
        "drafts": [],
        "verdicts": [],
    }


def errored(source: str = "case-1", max_refines: int = MAX_REFINES) -> dict:
    """A record of a case whose run raised, as ``build_error_record`` writes it.

    It has ``error`` and ``config``, but no ``terminal_state``, ``drafts`` or ``verdicts``.
    """
    return {
        "source": source,
        "error": {"type": "RefusalError", "message": "stub"},
        "config": {"max_refines": max_refines},
    }


def first_ten() -> list[dict]:
    """The first installment rebuilt from ``FIRST_TEN``, as ``case-1`` to ``case-10``."""
    return [case(rounds, f"case-{n}") for n, rounds in enumerate(FIRST_TEN, start=1)]


# --------------------------------------------------------------------------------------
# Your tests  [yours] -- what D34 asks to cover. Add
#   from curator.eval.convergence import concern_trend, section_repeats
# once the two functions exist.
#   concern_trend    one case worked out by hand first; down + flat + up == the number of
#                    review-to-review steps; first_ten()
#   section_repeats  one case worked out by hand first; a first review counts nowhere; all
#                    five sections present even at zero; first_ten() against the table in D34
#   both             adding not_case() and errored() records changes nothing
# --------------------------------------------------------------------------------------


def test_concern_trend_first_ten():
    """The first installment's ten cases, worked out by hand."""
    concern_trends = [concern_trend([case]) for case in first_ten()]
    assert concern_trends[0] == {"down": 2, "flat": 0, "up": 2}
    assert concern_trends[1] == {"down": 2, "flat": 0, "up": 0}
    assert concern_trends[2] == {"down": 2, "flat": 1, "up": 1}
    assert concern_trends[3] == {"down": 2, "flat": 0, "up": 2}
    assert concern_trends[4] == {"down": 2, "flat": 0, "up": 0}
    assert concern_trends[5] == {"down": 1, "flat": 2, "up": 1}
    assert concern_trends[6] == {"down": 1, "flat": 1, "up": 2}
    assert concern_trends[7] == {"down": 1, "flat": 0, "up": 0}
    assert concern_trends[8] == {"down": 2, "flat": 0, "up": 1}
    assert concern_trends[9] == {"down": 2, "flat": 1, "up": 1}
    total = concern_trend(first_ten())
    assert total == {"down": 17, "flat": 5, "up": 10}
    assert concern_trend(first_ten() + [not_case(), errored()]) == total
    assert sum(total.values()) == sum(len(case["verdicts"]) - 1 for case in first_ten())


def test_section_repeats():
    assert section_repeats([first_ten()[8]]) == {
        CPQ: {"repeated": 0, "new": 0},
        DR: {"repeated": 3, "new": 0},
        FD: {"repeated": 0, "new": 0},
        HAL: {"repeated": 0, "new": 0},
        SF: {"repeated": 0, "new": 1},
    }

    assert section_repeats([first_ten()[4]]) == {
        CPQ: {"repeated": 0, "new": 0},
        DR: {"repeated": 0, "new": 1},
        FD: {"repeated": 0, "new": 0},
        HAL: {"repeated": 0, "new": 0},
        SF: {"repeated": 0, "new": 0},
    }

    assert section_repeats(first_ten()) == {
        DR: {"repeated": 36, "new": 1},
        CPQ: {"repeated": 8, "new": 4},
        SF: {"repeated": 2, "new": 8},
        FD: {"repeated": 3, "new": 5},
        HAL: {"repeated": 0, "new": 3},
    }

    assert section_repeats(first_ten() + [not_case(), errored()]) == section_repeats(first_ten())
