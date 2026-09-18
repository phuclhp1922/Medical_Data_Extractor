"""The shared vocabulary of the pipeline: what each role is allowed to say.

Kept apart from ``parsing.py`` deliberately (D3). Everything touches these types -- prompts,
nodes, routers, the run records T7 writes, and ``GradeThresholds.passes`` in ``config.py`` --
so importing one must stay cheap. ``parsing.py`` imports ``openai`` because its whole job is
catching SDK failures, and a module is an all-or-nothing import: putting the models there
would drag the SDK into a router that only wants to read one score.

**No model the LLM fills in carries a default or a ``| None``.** Under OpenAI strict
structured outputs every property is forced into ``required`` regardless of defaults, and a
``None`` default is stripped before the schema is sent (D11, verified against ``openai
3.3.1``). So a default can never reach the model, and ``| None`` is not a convenience -- it is
a claim that "no value" is a legitimate answer, which for these models it never is. A field
that cannot be absent cannot be silently defaulted downstream, which is the whole point:
learning record 0002.

The error taxonomy at the bottom is the one exception, and it is not really one: those classes
are built by our own code, never by the model, so D11 does not govern them. Where one of them
does allow ``None`` (``RefusalError.refusal_text``), its constructor enforces exactly when.
"""

from enum import StrEnum

from pydantic import BaseModel

# --------------------------------------------------------------------------------------
# The editor's controlled vocabulary  (D12)
# --------------------------------------------------------------------------------------


class ChecklistSection(StrEnum):
    """Which item of the editor's audit a concern belongs to.

    The five values are the checklist headings A-E in ``prompts/editor.py``, and nothing else.
    They name *where the audit failed*, not what kind of fault it was. A fault vocabulary would
    have to be invented now, having observed zero flags in 51 cases (PLAN 2.6) -- so the
    taxonomy is deferred to Phase 3, where it can be coded from the explanations this schema
    collects. The section is the coarse, countable half; ``Concern.explanation`` carries the
    specifics.

    There is no ``NONE`` member. An empty ``EditorVerdict.concerns`` already means the draft is
    clean, and a ``NONE`` member would make ``[NONE, SOURCE_FIDELITY]`` constructible -- a
    state every consumer would then have to handle.

    ``SOURCE_FIDELITY`` and ``HALLUCINATIONS`` overlap, and that is accepted (D13). An invented
    lab value fails checklist A *and* checklist E, so the editor may label it either way.
    Deliberate, because the enum mirrors the audit the editor actually performs -- but it means
    some human-model disagreement in Phase 2 will be vocabulary rather than judgement, and the
    κ that results must be reported with that stated.
    """

    SOURCE_FIDELITY = "SOURCE_FIDELITY"
    CASE_PRESENTATION_QUALITY = "CASE_PRESENTATION_QUALITY"
    DIAGNOSTIC_REASONING = "DIAGNOSTIC_REASONING"
    FINAL_DIAGNOSIS = "FINAL_DIAGNOSIS"
    HALLUCINATIONS = "HALLUCINATIONS"


class Concern(BaseModel):
    """One thing the editor found wrong: which audit item failed, and what it saw.

    ``explanation`` is required, never ``str | None``. It is the reason this design exists: a
    closed label alone cannot say *what* was wrong, and a concern raised with nothing said is a
    bare label again. Under strict mode the model cannot omit it (D11), so the editor must
    always account for the section it names.
    """

    section: ChecklistSection
    explanation: str


class EditorVerdict(BaseModel):
    """The editor's answer for one draft.

    ``concerns`` is never ``list[Concern] | None``. A nullable list gives "clean" two spellings,
    ``[]`` and ``null``, and ``null`` is precisely how a non-answer would enter the record --
    the mechanism that recorded refusals as approvals in the notebook (learning record 0002).
    An empty list is the only way to say the draft passed.

    There is no ``satisfactory`` field. It is derivable as ``not verdict.concerns``, and a
    stored copy can disagree with the list it claims to summarise.

    A refusal or a truncated response is not an ``EditorVerdict`` at all -- those are the typed
    errors below, so no caller can mistake one for a verdict.
    """

    concerns: list[Concern]


# --------------------------------------------------------------------------------------
# What the extractor and the grader produce
# --------------------------------------------------------------------------------------


class Draft(BaseModel):
    """A teaching case as drafted by the extractor.

    The field names mirror the extractor's output blocks in ``prompts/extractor.py`` and the
    ``generated_*`` parameters of ``render_retry``. Keep the three in step: a rename here that
    is not made there gives T5 a translation layer for no reason.
    """

    think: str
    image_finding: str
    case_prompt: str
    reasoning_points: str
    reasoning_narrative: str
    final_diagnosis: str


class Grading(BaseModel):
    """The grader's verdict on one case report.

    ``think`` is declared first on purpose. Field order is generation order under structured
    outputs, so the reasoning is produced before any judgement it is supposed to support --
    including ``is_case_report``, which is itself a judgement on a borderline article. Declared
    last, it could only justify answers already committed to (D13).

    ``is_case_report`` replaces the rubric's old instruction to write ``"NA"`` into all six
    scores for a non-case-report -- a seventh output shape nothing downstream could represent
    (D5, defect 2). One flag makes disagreement between the six impossible.

    **The four scores are undefined when ``is_case_report`` is false, and must not be read.**
    Strict mode requires them, so the grader will emit four numbers for an article with no
    patient in it; they are an artefact of the format, not measurements. Every caller reads
    ``is_case_report`` first. ``GradeThresholds.passes`` gains that guard in T3 step 7 -- it
    does **not** have it yet -- and without it a chapter introduction scoring 3/3/3/3 clears
    every threshold and is sent on to the extractor. They are typed ``int`` rather than
    ``int | None`` so that absence is not reintroduced into the type: one shape, one guard,
    instead of a null check at every call site (D13).

    ``differential_diagnosis_present`` and ``final_diagnosis_present`` were called ``*_score``
    in the notebook, which is how ``int(...)`` came to sit beside ``== "Yes"`` in one
    conjunction. They are booleans and now say so. The grader prompt's ``<..._score>`` tags
    keep the old spelling: the rubric is frozen (D5), and the schema carries the contract now.
    """

    think: str
    is_case_report: bool

    case_presentation_score: int
    integrative_reasoning_score: int
    transparency_score: int
    images_usefulness_score: int
    differential_diagnosis_present: bool
    final_diagnosis_present: bool


# --------------------------------------------------------------------------------------
# Error taxonomy  (D14-D17)
# --------------------------------------------------------------------------------------


class Role(StrEnum):
    """Which pipeline role was running when a curation failure happened.

    Carried on every ``CurationError`` as ``role``. The same failure means different things at
    different stages -- a refused grading loses a case before it starts, a refused edit loses a
    draft already paid for -- so the run record needs to say where. A closed set, because Phase 2
    counts failures by role and three spellings of "editor" would make three rows (D15).
    """

    EXTRACTOR = "EXTRACTOR"
    EDITOR = "EDITOR"
    GRADER = "GRADER"


class StoppedBy(StrEnum):
    """Who stopped a refused output: the model itself, or the provider's content filter.

    Carried on ``RefusalError`` as ``stopped_by``. The distinction does not change what happens
    to the case -- either way it is skipped and recorded -- but it decides what to fix when many
    cases are skipped. A model refusal can be addressed by rewording the prompt; a provider
    filter cannot be reworded past, only avoided by changing model or provider (D14). This
    corpus is medical, and injuries, overdoses and abuse are exactly what filters misfire on.
    """

    MODEL = "MODEL"
    PROVIDER = "PROVIDER"


class CurationError(Exception):
    """Base of every way a model call can fail to produce a usable answer.

    **Never raised directly -- only its subclasses are.** It exists to be *caught*: one
    ``except CurationError`` can skip a case and record why, whatever the cause. A bare
    ``CurationError`` would tell that catcher nothing about which failure occurred, which is the
    one generic error this taxonomy replaces.

    These are expected outcomes, not bugs, so they are recorded rather than allowed to crash
    the run. A case that fails must still leave a record: "skip" always means skip *and record*,
    never vanish (D14). Everything that record needs from every failure lives here.

    Args:
        message: Human-readable summary, passed to ``Exception``.
        case_id: The case directory name, as ``figures.resolve_corpus`` keys it (D15).
        role: Which role was running.
    """

    def __init__(self, *, message: str, case_id: str, role: Role):
        super().__init__(message)
        self.case_id = case_id
        self.role = role


class RefusalError(CurationError):
    """The output was refused -- by the model, or by the provider's content filter.

    **A refusal is not an approval.** The notebook recorded an empty or refused editor response
    as ``flags="NONE"``, indistinguishable from an editor that read the draft and found nothing
    wrong -- the third confound behind the 0-of-51 (PLAN 2.6). As an exception, a refusal cannot
    take the shape of a verdict at all.

    Both kinds of stop are this one class because the caller treats them identically: no retry,
    skip, record (D14). Retrying cannot help -- the same content is refused again.

    ``refusal_text`` is ``None`` **exactly when** ``stopped_by`` is ``PROVIDER``: a filtered
    output was cut off before the model wrote anything, so no text exists. ``None`` rather than
    ``""``, because an empty string is a plausible value and would pass for a real, empty
    refusal. The constructor raises ``ValueError`` if the two disagree, so a model refusal whose
    reason was lost cannot be built (D17). That is a ``ValueError`` and not a ``CurationError``
    on purpose: a contradictory refusal is a bug in our code, not a curation outcome, and it
    should crash rather than be quietly recorded.

    Args:
        role: Which role was running.
        message: Human-readable summary.
        case_id: The case directory name.
        refusal_text: The model's own words when it refused; ``None`` if the provider stopped it.
            Kept because it says *what* to reword, and refusals do not reliably reproduce on a
            re-run (D17).
        stopped_by: Whether the model or the provider stopped the output.

    Raises:
        ValueError: If ``refusal_text`` is ``None`` for a model refusal, or present for a
            provider stop.
    """

    def __init__(
        self,
        *,
        role: Role,
        message: str,
        case_id: str,
        refusal_text: str | None,
        stopped_by: StoppedBy,
    ):
        if stopped_by == StoppedBy.MODEL and refusal_text is None:
            raise ValueError("refusal_text must be provided when stopped by MODEL")
        elif stopped_by == StoppedBy.PROVIDER and refusal_text is not None:
            raise ValueError("refusal_text must be None when stopped by PROVIDER")

        super().__init__(message=message, case_id=case_id, role=role)
        self.refusal_text = refusal_text
        self.stopped_by = stopped_by


class TruncationError(CurationError):
    """The output was cut off because it reached the token budget.

    The SDK reports this as an incomplete response with reason ``max_output_tokens``. Unlike a
    refusal, this one *can* be fixed by retrying -- with a larger budget -- because nothing about
    the content was objected to. The notebook never checked for it: the object was silently
    absent.

    The provider's *other* incomplete reason, ``content_filter``, is deliberately **not** this
    class, even though it also cuts the output short. Retrying a filtered output with more tokens
    is filtered again every time, spending rate limit on a retry that cannot succeed; it is a
    ``RefusalError`` (D14).

    Must be detected **before** schema validation. Strict mode guarantees valid JSON only if the
    model gets to finish, so a truncated response is also broken JSON -- checked in the wrong
    order, every truncation would be misreported as a ``SchemaError``.

    Takes exactly the arguments of ``CurationError``.
    """


class SchemaError(CurationError):
    """The model finished, and its output still did not match the schema.

    Under strict structured outputs this should almost never happen: the provider constrains
    decoding to the schema (D11). So when it does, it is evidence that the constraint was not
    applied as assumed -- strict mode not enabled, or not enforced -- and worth investigating
    rather than simply retrying. Everything needed to investigate is kept.

    It is only a schema error if the output was *complete*; a truncated output is a
    ``TruncationError`` and must be ruled out first.

    The wrapper raises this ``from`` the underlying Pydantic ``ValidationError``, so the precise
    field-level failure is on ``__cause__`` and is not duplicated as a field here.

    Args:
        role: Which role was running.
        message: Human-readable summary.
        case_id: The case directory name.
        raw_output: The exact text the model returned, unparsed.
    """

    def __init__(self, *, role: Role, message: str, case_id: str, raw_output: str):
        super().__init__(message=message, case_id=case_id, role=role)
        self.raw_output = raw_output
