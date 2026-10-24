"""
The funnel to count the cases ending in a terminal state.
"""

from curator.state import TerminalState


def funnel(records: list[dict]) -> dict[str, int]:
    """
    Count the number of cases ending in a terminal state.

    Args:
        records (list[dict]): A list of records, where each record stands for a case.

    Returns:
        dict[str, int]: A dictionary with the count of cases ending in each terminal state.
    """
    terminal_counts = {TerminalState.LOW_SCORE.value: 0,
                       TerminalState.NOT_CASE.value: 0,
                       TerminalState.HIT_MAX_REFINES.value: 0,
                       TerminalState.PASSED.value: 0,
                       "error": 0}


    for record in records:
        if "error" in record:
            terminal_counts["error"] += 1
        else:
            terminal_reason = record.get("terminal_state")
            terminal_counts[terminal_reason] += 1
    return terminal_counts
