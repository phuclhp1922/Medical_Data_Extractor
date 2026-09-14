"""Quality-grader prompt.

The grading criteria (thoroughness of case presentation, explicit differential diagnosis,
dependence on integrative reasoning, transparency of diagnostic reasoning, stated final
diagnosis) are adapted from the candidate-selection criteria of MedCaseReasoning (Wu et al.,
2025, arXiv:2505.11733).

Rubric text landed verbatim from ``notebooks/archive/03_curation_loop_langgraph.ipynb`` cell 12,
then split at the point where it first touches a variable (T2 step 2).

``RULES`` is invariant across all cases, so it is a module constant: it can be hashed for the
run record, asserted on in tests, and -- because it sits at the front of every payload -- reused
by the provider's prefix cache instead of being re-billed per case. ``render()`` holds the only
part that varies.

**This rubric is the project's measuring instrument**, not prose: it decides which cases enter
the dataset, and Phase 2 calibrates a human grader against this exact text. It is therefore
frozen at v1. Three defects were repaired first (T2 step 3), each one a place where the rubric
contradicted itself -- no criterion was reworded, reweighted, added or removed:

1. Two criteria were both numbered ``5.``; the second is now ``6.``. A model and a human reading
   an ambiguous list can resolve it differently, which would put a typo inside the agreement
   statistic the whole calibration rests on.
2. ``Additional Rules 1`` told the model to output ``"NA"`` for all scores when the article is
   not a case report -- a seventh output shape nothing downstream can represent. Deleted; T6
   gives that condition a schema field instead. An instruction is a hope, a schema is a
   guarantee.
3. The header said "output only the five XML tags" while the template declares six. The
   template wins, since the parser reads tags rather than counts. This one was not cosmetic:
   an obedient model emits five, the sixth score is absent, and ``GradeThresholds.passes``
   reads it as ``.get(key, 0)`` -- so a missing answer silently becomes a failed gate,
   indistinguishable from a genuine rejection. It is the argument for T3 making these fields
   required rather than defaulted.

Anything further -- criterion wording, what separates a 4 from a 5, the threshold values -- is
a *different* rubric, and comparing rubrics is a Phase 2 experiment run one variable at a time.
"""

from curator.prompts import fingerprint

RULES = """
    You are an expert medical educator tasked with evaluating case reports for their diagnostic-reasoning value.
    You will be given full, uncleaned text that has just been extracted from PDF via OCR. You will have to understand the text because some parts are splited due to layout.
    These images are passed to you together with the case.
    The goal is to check if this case report can be use like a diagnostic teaching cases from medical textbooks.

    CASE REPORT EVALUATION RUBRIC
    >> HOW TO USE
    1. Read the entire case once without scoring.
    2. Re-read, taking notes.
    3. Inside <think>...</think>, write the reasoning that leads you to each score.
    4. Output only the six XML tags shown after the rubric—nothing else.

    +-------------------------------------------------------------+
    | 1. THOROUGHNESS OF CASE PRESENTATION (1-5 points) |
    | Look for: HPI, past history, meds, allergies, vitals,   |
    | focused exam, labs, imaging, hospital course, outcome.  |
    | 1: Seriously deficient (identifiers only; no vitals)    |
    | 2: Major gaps (HPI + vitals OR exam, not both).         |
    | 3: Adequate (present but sketchy details).              |
    | 4: Very good (complete data, clear timeline).           |
    | 5: Exemplary (serial data & course, high quality).      |
    +-------------------------------------------------------------+
    | 2. EXPLICIT DIFFERENTIAL DIAGNOSIS (Yes / No)             |
    | >=2 plausible alternatives? If yes → "Yes"; else → "No".  |
    +-------------------------------------------------------------+
    | 3. DEPENDENCE ON INTEGRATIVE CLINICAL REASONING (1-5)     |
    | Measures need to combine >=2 data points (hx, labs, etc)  |
    | 1: Trivial: lone clue gives answer.                       |
    | 2: Minimal: one dominant clue.                            |
    | 3: Moderate: must merge TWO findings.                     |
    | 4: High: THREE+ clues; requires synthesis.                |
    | 5: Outstanding: stepwise, complex reasoning.              |
    +-------------------------------------------------------------+
    | 4. TRANSPARENCY OF DIAGNOSTIC REASONING PROCESS (1-5) |
    | 1: None (no rationale).                               |
    | 2: Superficial (lists w/o "why").                     |
    | 3: Adequate (brief pivots).                           |
    | 4: Detailed (stepwise, probabilities).                |
    | 5: Model (structured, addresses pitfalls).            |
    +-------------------------------------------------------------+
    | 5. USEFULNESS OF IMAGES (1-5)                                  |
    | 1: None (no usefulness or relatedness).                        |
    | 2: Low (very limited or marginal usefulness or relevance).     |
    | 3: Average (illustration or example, limited relevance).       |
    | 4: High (high usefulness and relevance, but can be omitted).   |
    | 5: Very high (cannot be omitted without serious consequences). |
    +-------------------------------------------------------------+
    | 6. STATED FINAL DIAGNOSIS (Yes / No)                  |
    | Is diagnosis clearly named? Yes → "Yes"; else → "No". |
    +-------------------------------------------------------------+

    Additional Rules:
    1. Think through whether the final diagnosis can be reasonably deduced from the case presentation when determining the
    educational value in #3.

    OUTPUT TEMPLATE (leave tags exactly as written)
    <think>
    ...your internal reasoning for each item...
    </think>

    <case_presentation_score>[1-5]</case_presentation_score>
    <differential_diagnosis_score>[Yes/No]</differential_diagnosis_score>
    <integrative_reasoning_score>[1-5]</integrative_reasoning_score>
    <transparency_score>[1-5]</transparency_score>
    <images_usefulness_score>[1-5]</images_usefulness_score>
    <final_diagnosis_score>[Yes/No]</final_diagnosis_score>
"""


#: Fingerprint of the frozen v1 rubric. T7 records this on every case, so a score can always be
#: traced to the exact rubric text that produced it.
RULES_VERSION = fingerprint(RULES)


def render(full_text: str) -> str:
    """Build the grader payload: the invariant rubric, then this case's text.

    Args:
        full_text (str): The uncleaned OCR text of one case report.

    Returns:
        str: The full grader prompt: the frozen v1 rubric, then this case's text.
    """
    return RULES + f"""
        SUPPLIED CASE REPORT
        <full_text>
        {full_text}
        </full_text>
    """
