"""Tests for ``graph.py``: where a case goes next, and how a run ends.

No network, no key, no corpus. Two things stand in for the world:

- ``LLM.parse`` is replaced by a ``FakeModel`` that answers from a script, so "the editor
  complains three times and then approves" is a list you write in the test.
- ``load_case`` is replaced by a stand-in that returns placeholder text and no figures, so the
  graph runs without ``extracted_data/``.

Everything else is real: the prompts are rendered, the nodes run, the routers route, and the
graph is the one ``build_graph()`` compiles.

**Note on authorship.** The agent wrote the wiring in ``graph.py`` and this harness; you wrote
``review_route``'s rule. The claims below are about that rule and about how runs end, and every
``assert`` is yours. Unlike ``test_llm.py``, the docstrings here state the claim and what is
available -- not the lines to type.

There are two kinds of test in this file, and it is worth noticing which is which:

- **Router tests** call ``review_route`` or ``grade_route`` directly with a hand-built state. No
  graph, no fake model. They are about the rule.
- **Run tests** push a case through the compiled graph with ``run_case``. They are about the
  rule and the wiring agreeing with each other.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import curator.graph as graph_module
from curator.config import Configuration
from curator.graph import (  # noqa: F401
    EXTRACT_CASE,
    TERMINALS,
    build_graph,
    grade_route,
    review_route,
)
from curator.llm import LLM
from curator.schemas import (
    ChecklistSection,
    Concern,
    Draft,
    EditorVerdict,
    Grading,
    Role,
)
from curator.state import TerminalState  # noqa: F401

#: Placeholder for the case text. Never real corpus text: the source is copyrighted.
SOURCE_TEXT = "ZZQQ_CASE_SENTINEL"


# --------------------------------------------------------------------------------------
# Building blocks: gradings and verdicts to script with
# --------------------------------------------------------------------------------------


def grading(*, score: int = 3, is_case_report: bool = True) -> Grading:
    """A grading with all four scores set to ``score``.

    The default thresholds are 3 across the board, so ``grading()`` passes them and
    ``grading(score=1)`` does not.
    """
    return Grading(
        think="stub",
        is_case_report=is_case_report,
        case_presentation_score=score,
        integrative_reasoning_score=score,
        transparency_score=score,
        images_usefulness_score=score,
        differential_diagnosis_present=True,
        final_diagnosis_present=True,
    )


def clean() -> EditorVerdict:
    """A verdict with no concerns -- the editor approves."""
    return EditorVerdict(concerns=[])


def concerned(*sections: ChecklistSection) -> EditorVerdict:
    """A verdict with one concern per section named.

    With no arguments it raises a single ``SOURCE_FIDELITY`` concern, which is enough for any
    test that only cares *whether* the editor complained.
    """
    sections = sections or (ChecklistSection.SOURCE_FIDELITY,)
    return EditorVerdict(
        concerns=[Concern(section=section, explanation="stub") for section in sections]
    )


# --------------------------------------------------------------------------------------
# The fake model
# --------------------------------------------------------------------------------------


class FakeModel:
    """Stands in for every model call in a run, answering from a script.

    Args:
        grading (Grading): What the grader returns.
        verdicts (list[EditorVerdict]): What the editor returns, one per review, in order.
        then (EditorVerdict | None): What the editor returns once ``verdicts`` is used up.
            Leave it ``None`` and a review past the end of the script raises -- so a loop that
            runs longer than you scripted fails loudly instead of spinning.

    Attributes:
        roles (list[Role]): Which role made each call, in the order the calls happened. The
            whole shape of a run can be read off this list.
    """

    def __init__(
        self,
        *,
        grading: Grading,
        verdicts: list[EditorVerdict] | None = None,
        then: EditorVerdict | None = None,
    ) -> None:
        self._grading = grading
        self._verdicts = list(verdicts or [])
        self._then = then
        self.roles: list[Role] = []

    def count(self, role: Role) -> int:
        """How many calls ``role`` made."""
        return self.roles.count(role)

    def answer(self, schema: type, role: Role):
        self.roles.append(role)
        if schema is Grading:
            return self._grading
        if schema is Draft:
            # Numbered, so two drafts in a run are tellable apart.
            n = self.count(Role.EXTRACTOR)
            return Draft(
                think="stub",
                image_finding="stub",
                case_prompt=f"draft {n}",
                reasoning_points="stub",
                reasoning_narrative="stub",
                final_diagnosis="stub",
            )
        if schema is EditorVerdict:
            if self._verdicts:
                return self._verdicts.pop(0)
            if self._then is not None:
                return self._then
            raise AssertionError("the editor was asked for more reviews than the test scripted")
        raise AssertionError(f"unexpected schema {schema!r}")


# --------------------------------------------------------------------------------------
# Running things
# --------------------------------------------------------------------------------------


def config(**overrides) -> Configuration:
    """A ``Configuration`` with defaults, pacing off, and any field overridden.

    ``config(max_refines=1)`` is the usual call.
    """
    return replace(Configuration.from_env({}), pause_seconds=0.0, **overrides)


def route_review(verdicts: list[EditorVerdict], *, max_refines: int = 3) -> str:
    """Call ``review_route`` on a state holding just these verdicts. No graph involved.

    Returns the name of the node the router chose, comparable with ``EXTRACT_CASE`` and
    ``TERMINALS[TerminalState....]``.
    """
    runtime = SimpleNamespace(context=config(max_refines=max_refines))
    return review_route({"verdicts": verdicts}, runtime)


def route_grade(result: Grading) -> str:
    """Call ``grade_route`` on a state holding just this grading. No graph involved."""
    return grade_route({"grading": result}, SimpleNamespace(context=config()))


@pytest.fixture
def run_case(monkeypatch):
    """Return a function that pushes one case through the real compiled graph.

    Usage: ``out = run_case(model, max_refines=1)``, where ``model`` is a ``FakeModel``.

    ``out`` is the final state: ``out["terminal_state"]``, ``out["drafts"]``,
    ``out["verdicts"]``, ``out["grading"]``. After the run, ``model.roles`` holds every call
    that was made.
    """

    def fake_load_case(state, runtime) -> dict:
        return {"source_text": SOURCE_TEXT, "figures": []}

    monkeypatch.setattr(graph_module, "load_case", fake_load_case)

    def _run(model: FakeModel, **overrides) -> dict:
        def fake_parse(llm, schema, *, input, case_id, role, **options):
            return model.answer(schema, role)

        monkeypatch.setattr(LLM, "parse", fake_parse)
        return build_graph().invoke({"source": "case-under-test"}, context=config(**overrides))

    return _run


# --------------------------------------------------------------------------------------
# The review router  [yours] -- use route_review(...); no fixture needed
# --------------------------------------------------------------------------------------


def test_a_clean_verdict_ends_the_case_as_passed():
    """One review, no concerns: the case is done and it passed."""
    route = route_review([clean()])
    assert route == TERMINALS[TerminalState.PASSED]


def test_concerns_with_retries_left_send_the_case_back():
    """One review with concerns, and the limit not reached: the case goes round again."""
    route = route_review([concerned()])
    assert route == EXTRACT_CASE


def test_concerns_at_the_limit_end_the_case():
    """Concerns on every review, and the retries used up: the case ends, and not as passed.

    This is the off-by-one D27 records. With ``max_refines=3`` the first draft is not a refine,
    so the question is exactly how many verdicts the state must hold before the router stops.
    Claim both sides of the edge: one verdict fewer still goes round again.
    """
    route_1 = route_review([concerned()] * 4)  # 4 concerns, max_refines=3
    route_2 = route_review([concerned()] * 3)  # 3 concerns, max_refines=3
    assert route_1 == TERMINALS[TerminalState.HIT_MAX_REFINES]
    assert route_2 == EXTRACT_CASE


def test_only_the_newest_verdict_decides():
    """Earlier complaints do not count against a case whose latest review is clean."""
    route = route_review([concerned(), clean()])
    assert route == TERMINALS[TerminalState.PASSED]


def test_zero_refines_means_one_attempt():
    """``max_refines=0``: a first draft with concerns is not sent back at all."""
    route = route_review([concerned()], max_refines=0)
    assert route == TERMINALS[TerminalState.HIT_MAX_REFINES]

# --------------------------------------------------------------------------------------
# The grade router  [yours] -- use route_grade(...)
# --------------------------------------------------------------------------------------


def test_a_low_grading_never_reaches_the_extractor():
    """A case below the thresholds ends at the low-score terminal; one above goes to extract."""
    route_1 = route_grade(grading(score=1))
    route_2 = route_grade(grading(score=5))
    assert route_1 == TERMINALS[TerminalState.LOW_SCORE]
    assert route_2 == EXTRACT_CASE


# --------------------------------------------------------------------------------------
# Whole runs  [yours] -- these take the run_case fixture
# --------------------------------------------------------------------------------------


def test_an_editor_that_always_complains_still_terminates(run_case):
    """T5's done-when (PLAN 2.3): the loop is bounded, and the record says why it stopped.

    Script an editor that never approves (``FakeModel(..., then=concerned())``) and pick a
    ``max_refines``. Claim how the run ended, and how many drafts and reviews it took to get
    there. The notebook had no cap: this run would not have ended.
    """
    run = run_case(
        FakeModel(
            grading=grading(score=5),
            verdicts=[concerned()] * 3,
            then=concerned(),
        ),
        max_refines=3,
    )
    assert run["terminal_state"] == TerminalState.HIT_MAX_REFINES
    assert len(run["drafts"]) == 4
    assert len(run["verdicts"]) == 4


def test_a_case_approved_first_time_is_one_draft_and_one_review(run_case):
    """The shortest passing run."""
    run = run_case(
        FakeModel(
            grading=grading(score=5),
            verdicts=[clean()],
        ),
        max_refines=3,
    )
    assert run["terminal_state"] == TerminalState.PASSED
    assert len(run["drafts"]) == 1
    assert len(run["verdicts"]) == 1


def test_a_case_fixed_on_the_second_try_keeps_both_drafts(run_case):
    """Complain once, then approve. The run passes, and nothing was overwritten on the way.

    The history is the point (D22): both drafts and both verdicts are still in the final state,
    oldest first. The fake numbers its drafts in ``case_prompt``, so order is checkable.
    """
    run = run_case(
        FakeModel(
            grading=grading(score=5),
            verdicts=[concerned(), clean()],
        ),
        max_refines=3,
    )
    assert run["terminal_state"] == TerminalState.PASSED
    assert len(run["drafts"]) == 2
    assert len(run["verdicts"]) == 2
    assert run["drafts"][0].case_prompt == "draft 1"
    assert run["drafts"][1].case_prompt == "draft 2"


def test_a_low_scoring_case_costs_one_call(run_case):
    """A case that fails grading stops there. ``model.roles`` shows what was and was not called."""
    model = FakeModel(grading=grading(score=1))
    run = run_case(model, max_refines=3)

    assert run["terminal_state"] == TerminalState.LOW_SCORE
    assert len(run["drafts"]) == 0
    assert len(run["verdicts"]) == 0
    assert run["grading"] is not None
    # The grader was called, but nothing else.
    assert model.roles == [Role.GRADER]

# One more, with no stub: T6's gap. Today a grading with ``is_case_report=False`` ends as
# ``low_score``, and ``TerminalState.NOT_CASE`` has no writer. Whether to write a test that pins
# today's behaviour, or to leave it until T6 changes it, is your call -- a test that asserts a
# known-wrong outcome has to be rewritten the day the bug is fixed.
