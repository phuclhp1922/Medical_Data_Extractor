"""The graph's state: what one case carries as it moves through the pipeline.

Three schemas, not one. ``InputState`` is what a caller may supply; ``CaseState`` is
everything the nodes keep between them. Wired as ``StateGraph(CaseState,
input_schema=InputState, context_schema=Configuration)``, which is what stops
``app.invoke({...})`` being an untyped dict a caller can put anything into -- verified: without
an ``input_schema`` a caller can set *any* field, ``terminal_state`` included, and the graph
starts half-finished with no error and no warning (``teaching/reference/langgraph-state.html``).

``TypedDict``, not Pydantic. These are containers for values that were already validated on the
way in -- ``Grading`` and ``EditorVerdict`` by ``parsing.py``, ``Figure`` by ``figures.py``.
Validating them a second time on every node write would buy nothing.

**Settings do not live here.** Model ids, thresholds and ``max_refines`` reach nodes through
``Runtime[Configuration]``, so state stays a record of *this case* rather than a mixture of
case data and run configuration.

**Image bytes do not live here either.** State holds ``Figure`` objects -- label, path, caption
-- and ``images.py`` decodes them into content parts at call time. T7 serialises this whole
object per case; base64 in a channel would write tens of megabytes of copyrighted figures into
every run record.

A field's comment says which node writes it. That is the useful thing to know about a channel:
two writers of one field is a design question, and ``terminal_state`` having exactly one is a
guarantee T7 depends on.
"""

# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------
# operator (for the reducer), Annotated + TypedDict, StrEnum, and the already-validated
# types this state carries: Figure from figures, Draft / EditorVerdict / Grading from schemas.

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

from curator.figures import Figure
from curator.schemas import Draft, EditorVerdict, Grading

# --------------------------------------------------------------------------------------
# How a case can stop
# --------------------------------------------------------------------------------------


class TerminalState(StrEnum):
    """Why a case left the graph. Written by ``finalize`` and by nothing else.

    One member per distinct way a case can end. Four are known now:

    - the grader scored it below ``GradeThresholds`` (PLAN 2.4)
    - the grader said it is not a case report at all -- T6
    - the refine loop hit ``max_refines`` without a clean verdict -- T5
    - it passed

    Every case must land in exactly one of these; T7's funnel test is precisely that partition,
    which is why "unset" must not be a member. A case still running has no ``terminal_state``
    key at all, and reading it raises ``KeyError`` -- that is the desired behaviour, not a gap
    to fill with a default (learning record 0002).

    Naming is yours. Prefer names that say what happened to the case, not where control went:
    two of these route to the same node.
    """
    LOW_SCORE = "low_score"
    NOT_CASE = "not_case"
    HIT_MAX_REFINES = "hit_max_refines"
    PASSED = "passed"


# --------------------------------------------------------------------------------------
# What a caller supplies  (D23)
# --------------------------------------------------------------------------------------


class InputState(TypedDict):
    """The graph's entry point: one case, named.

    One field -- the case directory name, as ``figures.resolve_corpus()`` already keys cases.
    ``load_case`` derives the directory from it via ``config.corpus_dir``. Carrying the path as
    a second field would let a caller pass a source and a path that disagree, and the run record
    would then name the wrong case; one field makes that unrepresentable (D23).
    """

    source: str


# --------------------------------------------------------------------------------------
# What the nodes keep between them
# --------------------------------------------------------------------------------------


class CaseState(TypedDict):
    """Everything one case accumulates. Grouped by which node writes each field.

    The two reducer channels are the substance of D22. Default channel behaviour is
    last-write-wins, so a four-pass case would end holding one verdict and one draft with the
    earlier three of each discarded -- exactly the loss PLAN 2.7 is about. ``Annotated[list[X],
    operator.add]`` appends instead, giving the flag trajectory across refines for free.

    With a reducer, a node returns **only the new item** (``{"verdicts": [v]}``); returning the
    whole list appends a second copy of everything already there.
    """

    # --- from the caller (mirrors InputState) ---
    source: str

    # --- written by load_case: the source text, and the figures (never bytes) ---
    source_text: str
    figures: list[Figure]

    # --- written by grade_case ---
    grading: Grading

    # --- the refine loop: two reducer channels (D22) ---
    #     extract_case appends a Draft; review_case appends an EditorVerdict, and owns
    #     refine_count (T5: max_refines = 3 means four extract passes).
    verdicts: Annotated[list[EditorVerdict], operator.add]
    drafts: Annotated[list[Draft], operator.add]

    refine_count: int  # how many times the case has been through the refine loop

    # --- written by finalize, and only by finalize ---
    terminal_state: TerminalState  # why the case left the graph
