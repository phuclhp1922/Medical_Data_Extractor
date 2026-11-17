"""Is the loop converging, or moving the goalposts? Two counts over consecutive reviews.

Phase 2 step 2 (PLAN.md; D34). Concerns in consecutive reviews of one case are matched by
``section`` alone: free, exact and repeatable, but coarse. DIAGNOSTIC_REASONING is flagged in
nearly every review of every stuck case, so a section match there cannot tell the same complaint
from a new one in the same place. Report the repeated share as an upper bound, and say so.
"""

from curator.schemas import ChecklistSection

SECTIONS = [section.value for section in ChecklistSection]


def concern_trend(records: list[dict]) -> dict[str, int]:
    """Count how often the number of concerns fell, held or rose from one review to the next.

    A case with n reviews gives n - 1 steps. Error records are skipped, and a case stopped at
    grading has no reviews, so it adds nothing.

    Args:
        records: A batch's records, one per case.

    Returns:
        ``{"down": n, "flat": n, "up": n}``: how many steps had fewer, the same, or more concerns
        than the review before.
    """

    result = {"down": 0, "flat": 0, "up": 0}

    for record in records:
        if "error" not in record and record["verdicts"]:
            for i in range(1, len(record["verdicts"])):
                prev_count = len(record["verdicts"][i - 1]["concerns"])
                curr_count = len(record["verdicts"][i]["concerns"])
                if curr_count < prev_count:
                    result["down"] += 1
                elif curr_count == prev_count:
                    result["flat"] += 1
                else:
                    result["up"] += 1

    return result


def section_repeats(records: list[dict]) -> dict[str, dict[str, int]]:
    """Count each section's concerns as repeated or new, against the review just before.

    Error records are skipped, and a case stopped at grading has no reviews, so it adds nothing.
    A first review has nothing before it, so its concerns are neither repeated nor new.

    Args:
        records: A batch's records, one per case.

    Returns:
        ``{section: {"repeated": n, "new": n}}``: for each of the five sections, how many
        concerns had that section flagged in the review just before (``"repeated"``) and how many
        did not (``"new"``). Every section is present, zeros included.
    """

    result = {section: {"repeated": 0, "new": 0} for section in SECTIONS}

    for record in records:
        if "error" not in record and record["verdicts"]:
            for i in range(1, len(record["verdicts"])):
                previous_set = {
                    concern["section"] for concern in record["verdicts"][i - 1]["concerns"]
                }
                verdict = record["verdicts"][i]

                for concern in verdict["concerns"]:
                    section = concern["section"]
                    if section in previous_set:
                        result[section]["repeated"] += 1
                    else:
                        result[section]["new"] += 1

    return result
