"""The pipeline's nodes: one function per step a case goes through.

A node takes the state and the runtime, and returns an **update** -- the keys it changed, not
the whole state. LangGraph merges that update into the channels declared in ``state.py``.
Two rules follow from how those channels are declared, and both are silent when broken:

- A key that is not declared in ``CaseState`` is **dropped without warning**. Nodes therefore
  read by indexing (``state["grading"]``), never ``.get``, so a lost write surfaces as a
  ``KeyError`` at the first reader instead of as a plausible default.
- ``verdicts`` and ``drafts`` are ``operator.add`` channels. A node appends by returning
  **only the new item** -- ``{"drafts": [draft]}``. Returning the accumulated list appends a
  second copy of everything already in it.

Nodes here hold pipeline logic only. Anything that waits, retries or holds a key lives in
``llm.py``; anything that decodes an image lives in ``images.py``; the wording of every request
lives in ``prompts/``. That division is what makes T5 fast -- stub ``LLM.parse`` and the whole
refine loop runs with no network and no sleeping -- and it is why no node contains a
``time.sleep``, unlike all four of their notebook originals.

Settings arrive through ``Runtime[Configuration]`` rather than through state, so a run's
configuration stays separable from what the run produced.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from langgraph.runtime import Runtime
from openai.types.responses import ResponseInputParam, ResponseInputTextParam

from curator.config import Configuration
from curator.figures import CorpusError, Figure, resolve_case
from curator.images import as_content_parts
from curator.llm import JUDGE_API_KEY_ENV, LLM
from curator.prompts import editor, extractor, grader
from curator.schemas import Draft, EditorVerdict, Grading, Role
from curator.state import CaseState, TerminalState


def _user_message(prompt: str, figures: Sequence[Figure]) -> ResponseInputParam:
    """Assemble one user turn: the prompt, then each figure labelled and shown.

    Every role sends the same shape -- instructions first, images after -- so this is written
    once. Order matters twice over: the model reads the instructions before the evidence they
    are about, and providers cache on *prefix*, so the invariant rubric leading each payload is
    what makes a cache hit possible across all 93 cases (see ``tests/test_prompts.py``).

    Args:
        prompt (str): The rendered prompt for this role, from ``prompts/``.
        figures (Sequence[Figure]): The case's figures. May be empty.

    Returns:
        ResponseInputParam: A single-element input list holding one user message.
    """
    return [
        {
            "role": "user",
            "content": [
                ResponseInputTextParam(type="input_text", text=prompt),
                *as_content_parts(figures),
            ],
        }
    ]


def _source_text(case_dir: Path) -> str:
    """Read the case's extracted text.

    MinerU writes one Markdown rendering of the PDF per case, next to the
    ``_content_list.json`` that ``figures.resolve_case()`` reads. The two are different views
    of the same extraction: the Markdown is the prose, the JSON is the structure.

    Args:
        case_dir (Path): The case directory -- the parent of ``auto/``, not ``auto/`` itself.

    Returns:
        str: The Markdown as written, unmodified.

    Raises:
        CorpusError: No Markdown file, or more than one. Either means the directory is not the
            MinerU output it is being treated as, and guessing which file to read would make
            the run's provenance unreproducible.
    """
    candidates = sorted((case_dir / "auto").glob("*.md"))
    if len(candidates) != 1:
        raise CorpusError(
            f"expected exactly one .md in {case_dir / 'auto'}, found {len(candidates)}"
        )
    return candidates[0].read_text(encoding="utf-8")


def load_case(state: CaseState, runtime: Runtime[Configuration]) -> dict:
    """Read one case off disk: its text, and its figures with labels attached.

    The only node that touches the corpus, and the place PLAN 2.1 is fixed. The notebook
    globbed all 219 ``.jpg`` under the corpus and sent 77 tables and decorations among them,
    unlabelled -- which is why "cite Fig 24.1" was an instruction no model could obey.
    ``figures.resolve_case()`` returns the real figures with their labels instead.

    Figures are carried as ``Figure`` objects. Nothing here decodes an image: ``images.py``
    does that at call time, so no base64 ever enters a channel or a T7 record.

    Args:
        state (CaseState): Carries ``source`` -- the case directory name.
        runtime (Runtime[Configuration]): Supplies ``corpus_dir``.

    Returns:
        dict: ``source_text``, ``figures``, and ``refine_count`` set to 0 so the refine loop
        starts from a known count. ``verdicts`` and ``drafts`` are not initialised: an
        ``operator.add`` list channel starts empty on its own (probe E11).

    Raises:
        CorpusError: The directory is not readable as MinerU output.
    """
    case_dir = runtime.context.corpus_dir / state["source"]
    source_text = _source_text(case_dir)
    figures = resolve_case(case_dir)

    return {
        "source_text": source_text,
        "figures": figures,
        "refine_count": 0,
    }


def grade_case(state: CaseState, runtime: Runtime[Configuration]) -> dict:
    """Score the source case report against the frozen rubric, before anything is generated.

    The gate that decides whether a case is worth extracting from. It judges the *textbook's*
    case, not ours -- so it runs once, ahead of the refine loop, and is never repeated.

    **It does not decide whether the case passed.** ``GradeThresholds.passes`` is a pure
    function of the grading, so ``grade_route`` can compute it when it needs it. A
    ``passed_grading`` field, as the notebook had, would be a second copy of a fact already in
    the state -- free to drift, and one more thing T7 would have to record and reconcile.

    **Nor does it catch the typed errors.** A refusal, a truncation or a schema failure ends
    the run loudly. The notebook turned a refusal into ``passed_grading: False`` -- a
    non-answer recorded as a judgement, and the third confound behind the 0-of-51 figure
    (PLAN 2.6). Until there is a terminal state that means "we never got an answer", crashing
    is the honest option: 93 cases is a cheap rerun, and a silent miscount is not.

    Args:
        state (CaseState): Needs ``source``, ``source_text`` and ``figures``.
        runtime (Runtime[Configuration]): Supplies the model id, pacing and retry policy.

    Returns:
        dict: ``grading`` alone. A case that is not a case report is still only a grading
        here: acting on ``is_case_report`` is ``grade_route``'s job, and ``terminal_state`` is
        written by ``finalize`` and nothing else.

    Raises:
        RefusalError: The grader refused, or a content filter stopped the output.
        TruncationError: The grading hit the output limit part-written.
        SchemaError: The output did not match ``Grading``.
    """
    llm = LLM(
        model=runtime.context.curator_model,
        pause_seconds=runtime.context.pause_seconds,
        request_retries=runtime.context.request_retries,
        temperature=runtime.context.temperature,
    )
    grading = llm.parse(
        Grading,
        input=_user_message(grader.render(state["source_text"]), state["figures"]),
        case_id=state["source"],
        role=Role.GRADER,
    )

    return {"grading": grading}


def _editor_findings(verdict: EditorVerdict) -> tuple[str, str]:
    """Flatten one verdict into the two strings the retry prompt interpolates.

    ``prompts.extractor.render_retry`` takes ``flags`` and ``editor_comments`` as text, because
    that is the shape the prompt was written around and the rubric is frozen (D5). The verdict
    is structured, so something has to flatten it; doing that here keeps ``prompts/`` free of
    logic and keeps the structure in the state, where T7 records it.

    Args:
        verdict (EditorVerdict): The verdict the extractor is answering.

    Returns:
        tuple[str, str]: ``flags`` -- the failed section names, comma-separated, which is the
        shape the notebook's prompt showed the model -- and ``editor_comments``, one
        ``SECTION: explanation`` line per concern. The section is repeated in both because a
        bare label cannot say what was wrong and a bare explanation cannot say where (D11's
        reasoning, applied to the way back).
    """
    flags = ", ".join(concern.section.value for concern in verdict.concerns)
    comments = "\n".join(
        f"{concern.section.value}: {concern.explanation}" for concern in verdict.concerns
    )
    return flags, comments


def extract_case(state: CaseState, runtime: Runtime[Configuration]) -> dict:
    """Write a teaching case from the source report -- first try, or a re-try after review.

    The one node that runs more than once, and the only writer of ``drafts``. Which prompt it
    sends is decided by the state, not by a flag a caller passes: an empty ``verdicts`` means
    nothing has reviewed anything yet, so this is the first attempt. Every later pass exists
    *because* of a verdict, and answers the newest one -- ``verdicts[-1]``.

    Both prompts carry the figures, not just the text. The image findings are the point of the
    dataset, and the retry has to be able to correct them (PLAN 2.1).

    Args:
        state (CaseState): Needs ``source``, ``source_text``, ``figures``, ``verdicts`` and
            ``drafts``.
        runtime (Runtime[Configuration]): Supplies the model id, sampling and retry policy.

    Returns:
        dict: The new ``draft`` appended to ``drafts``, and ``refine_count`` **derived** as
        ``len(verdicts)`` rather than incremented. The two can then never drift: a count kept
        by ``+= 1`` is a second record of how many verdicts there have been, and T5's cap is
        only meaningful if that number is the true one.

    Raises:
        RefusalError: The extractor refused, or a content filter stopped the output.
        TruncationError: The draft hit the output limit part-written.
        SchemaError: The output did not match ``Draft``.
    """
    if state["verdicts"]:
        previous = state["drafts"][-1]
        flags, editor_comments = _editor_findings(state["verdicts"][-1])
        prompt = extractor.render_retry(
            state["source_text"],
            previous.think,
            previous.image_finding,
            previous.case_prompt,
            previous.reasoning_points,
            previous.reasoning_narrative,
            previous.final_diagnosis,
            flags,
            editor_comments,
        )
    else:
        prompt = extractor.render(state["source_text"])

    llm = LLM(
        model=runtime.context.curator_model,
        pause_seconds=runtime.context.pause_seconds,
        request_retries=runtime.context.request_retries,
        temperature=runtime.context.temperature,
    )

    draft = llm.parse(
        Draft,
        input=_user_message(prompt, state["figures"]),
        case_id=state["source"],
        role=Role.EXTRACTOR,
    )

    return {
        "drafts": [draft],
        "refine_count": len(state["verdicts"]),
    }


def review_case(state: CaseState, runtime: Runtime[Configuration]) -> dict:
    """Audit the newest draft against the checklist, and report what is wrong with it.

    **The one role that must not be the curator model.** The editor reads text the extractor's
    family wrote, so running both on one model is the self-enhancement setup PLAN 2.6 names as a
    suspect behind 0 flags in 51 cases. It therefore builds its ``LLM`` from ``judge_model``,
    ``JUDGE_API_KEY`` and ``judge_base_url`` -- a different family, a different key, and
    usually a different vendor (D25).

    **It reports; it does not decide.** ``EditorVerdict`` carries concerns and nothing else:
    "satisfactory" is ``not verdict.concerns``, computed by ``review_route`` when it routes.
    The notebook stored that flag, and a stored flag can disagree with the list it summarises --
    which is how a refusal was once recorded as an approval (learning record 0002).

    The draft goes back as text, and the figures go with it: an editor asked whether an image
    finding is supported cannot answer without the image.

    Args:
        state (CaseState): Needs ``source``, ``source_text``, ``figures`` and ``drafts``.
        runtime (Runtime[Configuration]): Supplies the judge's model, endpoint and policy.

    Returns:
        dict: The new verdict appended to ``verdicts`` -- and nothing else. That list is the
        pipeline's record of how many review passes happened, which is why ``extract_case``
        reads its length instead of keeping a counter of its own.

    Raises:
        RefusalError: The editor refused, or a content filter stopped the output.
        TruncationError: The verdict hit the output limit part-written.
        SchemaError: The output did not match ``EditorVerdict``.
    """
    draft = state["drafts"][-1]

    llm = LLM(
        model=runtime.context.judge_model,
        pause_seconds=runtime.context.pause_seconds,
        request_retries=runtime.context.request_retries,
        temperature=runtime.context.judge_temperature,
        api_key_env=JUDGE_API_KEY_ENV,
        base_url=runtime.context.judge_base_url,
    )

    verdict = llm.parse(
        EditorVerdict,
        input=_user_message(
            editor.render(
                state["source_text"],
                draft.think,
                draft.image_finding,
                draft.case_prompt,
                draft.reasoning_points,
                draft.reasoning_narrative,
                draft.final_diagnosis,
            ),
            state["figures"],
        ),
        case_id=state["source"],
        role=Role.EDITOR,
    )

    return {"verdicts": [verdict]}


def finalize(reason: TerminalState) -> Callable[[CaseState, Runtime[Configuration]], dict]:
    """Build the node that ends a run for one reason.

    Called once per outcome at wiring time -- ``finalize(TerminalState.PASSED)`` and so on --
    so the graph holds four terminal nodes that share this single body. That is the whole point
    of the factory: ``terminal_state`` still has **one writer**, while the *value* comes from
    which edge was taken rather than from re-reading the state and guessing.

    The alternative was one ``finalize`` that re-derived the reason from ``grading``,
    ``verdicts`` and ``refine_count``. It was rejected because the reason is a decision a router
    already made, and reconstructing it is both a second copy of the routing rules and a trap:
    ``thresholds.passes()`` returns ``False`` for a non-case-report too, so a mis-ordered chain
    of conditions would file every ``NOT_CASE`` as ``LOW_SCORE``, and ``HIT_MAX_REFINES`` would
    come to mean "none of the other three matched". Same principle as ``EditorVerdict`` having
    no ``satisfactory`` field: keep the fact, do not derive it twice.

    This is also where T7 will persist the run record, because it is the one point every case
    reaches exactly once, whichever way it ends.

    Args:
        reason (TerminalState): Why runs ending at this node ended.

    Returns:
        Callable[[CaseState, Runtime[Configuration]], dict]: The node. Its ``__name__`` is set
        to ``finalize_<reason>`` so traces and error messages name the outcome, not a closure.
    """

    def _finalize(state: CaseState, runtime: Runtime[Configuration]) -> dict:
        """Record why this case left the graph.

        Args:
            state (CaseState): Not read today. T7 will record it from here.
            runtime (Runtime[Configuration]): Not read today.

        Returns:
            dict: The update. Yours to write -- one key, and ``reason`` is already in hand from
            the enclosing call, so nothing needs deciding here.
        """
        return {"terminal_state": reason}

    _finalize.__name__ = f"finalize_{reason.value}"
    return _finalize
