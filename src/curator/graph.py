"""The graph: which node runs when, and where a case can stop.

One case, one run. The shape is a straight line with one loop in it::

    START -> load_case -> grade_case -.grade_route.-> finalize_low_score  -> END
                                       \\-> extract_case -> review_case
                                                              |
                                          .review_route.-> extract_case (again)
                                                        \\-> finalize_passed -> END
                                                        \\-> finalize_hit_max_refines -> END

Three properties are worth stating, because each is a bug the notebook had:

- **Every path ends at a ``finalize_*`` node.** There is no edge to ``END`` that skips one, so
  ``terminal_state`` is set on every case that finishes, and a case can never be recorded with
  no reason for stopping. The notebook's loop simply stopped, and the reason lived in whatever
  the last print statement said.
- **Routers choose; they never write.** LangGraph forbids a router returning state, and the
  design leans on that: the reason a run ended is carried by *which* terminal node the router
  picked, not by a field some earlier node guessed at (D24, and ``nodes.finalize``).
- **The loop is bounded.** ``review_route`` compares ``len(verdicts)`` with ``max_refines``
  before sending a case back to ``extract_case``, so a case that never satisfies the editor
  leaves through ``finalize_hit_max_refines`` rather than spending money until something breaks.

``build_graph()`` is a function rather than a module-level singleton so that importing this
module compiles nothing and reads no environment.
"""

from langgraph.graph import END, START, StateGraph

from curator.config import Configuration
from curator.nodes import extract_case, finalize, grade_case, load_case, review_case
from curator.state import CaseState, InputState, TerminalState

#: Node names. Written once here and used for both registration and routing: a router that
#: returns a string no node answers to fails at compile time, which is the check these buy.
LOAD_CASE = "load_case"
GRADE_CASE = "grade_case"
EXTRACT_CASE = "extract_case"
REVIEW_CASE = "review_case"

#: One terminal node per outcome, named after the reason it records.
TERMINALS = {reason: f"finalize_{reason.value}" for reason in TerminalState}


def grade_route(state: CaseState, runtime) -> str:
    """Decide what happens to a graded case.

    **Minimal stub -- T6 completes it.** Today a case that fails the thresholds leaves as
    ``LOW_SCORE``, whatever the reason. That is wrong for one case in particular: a document the
    grader says is not a case report at all is not a low-scoring case report, and
    ``GradeThresholds.passes`` returns ``False`` for both, which is the confusion T6 exists to
    undo (PLAN 2.4). ``TerminalState.NOT_CASE`` is already declared and has no writer yet.

    Args:
        state (CaseState): Needs ``grading``.
        runtime: Supplies ``thresholds``.

    Returns:
        str: The name of the next node.
    """
    if runtime.context.thresholds.passes(state["grading"]):
        return EXTRACT_CASE
    return TERMINALS[TerminalState.LOW_SCORE]


def review_route(state: CaseState, runtime) -> str:
    """Decide whether the newest draft is done, needs another pass, or has run out of passes.

    **Minimal stub -- T5 completes it.** The cap is here already, deliberately: without it a
    case the editor never approves would loop until LangGraph's recursion limit stopped it, and
    a live run would pay for every pass on the way. What T5 owns is the *meaning* of the loop --
    whether the editor is raising the same concern each time or moving the goalposts (D22's
    second question), and what the record should say about exhaustion.

    The pass count is read here rather than stored: ``len(verdicts)`` is how many reviews have
    happened, so there is no counter to fall behind the list it summarises (D27). Note the
    arithmetic, because this is where the old ``refine_count`` hid an off-by-one: the first
    extract is not a *refine*, so the refines already done is one fewer than the verdicts in
    hand, and ``max_refines = 3`` therefore permits four extract passes -- what
    ``.env.example`` has always claimed it means.

    "Satisfactory" is computed here rather than stored: an empty ``concerns`` list is the only
    spelling of a clean draft (``EditorVerdict``), so there is nothing to disagree with.

    Args:
        state (CaseState): Needs ``verdicts``.
        runtime: Supplies ``max_refines``.

    Returns:
        str: The name of the next node.
    """
    if not state["verdicts"][-1].concerns:
        return TERMINALS[TerminalState.PASSED]
    refines_done = len(state["verdicts"]) - 1
    if refines_done >= runtime.context.max_refines:
        return TERMINALS[TerminalState.HIT_MAX_REFINES]
    return EXTRACT_CASE


def build_graph():
    """Wire and compile the pipeline.

    ``input_schema=InputState`` is what makes ``source`` the only key a caller may supply:
    anything else -- ``terminal_state`` included -- is dropped at the door rather than starting
    the run half-finished (D23, verified live).

    Returns:
        CompiledStateGraph: Ready to ``invoke({"source": ...}, context=Configuration(...))``.
    """
    builder = StateGraph(CaseState, input_schema=InputState, context_schema=Configuration)

    builder.add_node(LOAD_CASE, load_case)
    builder.add_node(GRADE_CASE, grade_case)
    builder.add_node(EXTRACT_CASE, extract_case)
    builder.add_node(REVIEW_CASE, review_case)
    for reason, name in TERMINALS.items():
        builder.add_node(name, finalize(reason))

    builder.add_edge(START, LOAD_CASE)
    builder.add_edge(LOAD_CASE, GRADE_CASE)
    builder.add_conditional_edges(
        GRADE_CASE, grade_route, [EXTRACT_CASE, TERMINALS[TerminalState.LOW_SCORE]]
    )
    builder.add_edge(EXTRACT_CASE, REVIEW_CASE)
    builder.add_conditional_edges(
        REVIEW_CASE,
        review_route,
        [
            EXTRACT_CASE,
            TERMINALS[TerminalState.PASSED],
            TERMINALS[TerminalState.HIT_MAX_REFINES],
        ],
    )
    for name in TERMINALS.values():
        builder.add_edge(name, END)

    return builder.compile()
