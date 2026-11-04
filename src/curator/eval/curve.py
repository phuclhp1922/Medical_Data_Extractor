"""The retry-budget curve: what each cap on refines would have bought, read from one frozen run.

The cap only acts when a case would go past it. Until then every routing decision is the same,
so the first ``k + 1`` drafts of a run at cap 4 are a run at cap ``k``, and one run holds the
curve for every lower cap (PLAN.md, Phase 2 step 1; D33). Higher caps are unknown, not zero: a
case that hit cap 4 has no fifth refine to read.
"""

from collections.abc import Mapping
from typing import Any

from curator.schemas import Role

#: The roles that run once per draft. Every other call -- the grading -- is paid at any cap.
_PER_DRAFT = (Role.EXTRACTOR.value, Role.EDITOR.value)


def calls_at_cap(record: Mapping[str, Any], cap: int) -> list[dict[str, Any]]:
    """The calls ``record``'s case would have paid for had the run stopped at ``cap`` refines.

    Every grading call, plus the extractor and editor calls of drafts 0 to ``cap``; a case that
    stopped sooner pays only for the drafts it has. ``CallUsage`` carries no round, so calls are
    matched to drafts by position: the nth EXTRACTOR and nth EDITOR entries are draft n.

    Raises:
        ValueError: ``cap`` is outside 0 to the run's own cap, or the record's calls do not line
            up one-to-one with its drafts and verdicts, so matching by position would be wrong.
    """
    run_cap = record["config"]["max_refines"]
    if not 0 <= cap <= run_cap:
        raise ValueError(f"cap {cap} is outside 0..{run_cap}, the caps this run can answer for")

    seen = {role: 0 for role in _PER_DRAFT}
    calls = []
    for call in record["usage"]:
        role = call["role"]
        if role in seen:
            seen[role] += 1
            if seen[role] > cap + 1:
                continue
        calls.append(call)

    expected = {
        Role.EXTRACTOR.value: len(record["drafts"]),
        Role.EDITOR.value: len(record["verdicts"]),
    }
    if seen != expected:
        raise ValueError(
            f"{record['source']}: calls per role {seen} do not match the drafts and verdicts"
            f" {expected}"
        )
    return calls


def first_clean_draft(record: dict) -> int | None:
    """The index of the first draft in ``record`` that had no concerns, or None if none did."""
    for i, verdict in enumerate(record["verdicts"]):
        if len(verdict["concerns"]) == 0:
            return i
    return None


def retry_budget_curve(records: list[dict]) -> dict[int, dict[str, int]]:
    """The retry-budget curve for the given frozen run.

    Args:
        records: The frozen run's records.

    Returns:
        A dict mapping each cap to a dict with keys:
            - ``entered``: the number of cases that would have been run at that cap
            - ``passed``: the number of cases that would have passed at that cap
            - ``hit_cap``: the number of cases that would have hit the cap without passing
    """

    curve = {}
    first_pass = {}

    cap = records[0]["config"]["max_refines"]

    for record in records:
        if "error" not in record and record["verdicts"]:
            first_pass[record["source"]] = first_clean_draft(record)

    for i in range(cap + 1):
        curve[i] = {
            "entered": len(first_pass),
            "passed": sum(1 for _, first in first_pass.items() if first is not None and first <= i),
            "hit_cap": sum(1 for _, first in first_pass.items() if first is None or first > i),
        }

    return curve
