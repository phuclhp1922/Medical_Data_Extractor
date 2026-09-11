"""Quality-grader prompt.

Landed verbatim from ``notebooks/archive/03_curation_loop_langgraph.ipynb`` cell 12.
Nothing here is edited yet: T2 step 1 is a pure relocation so that the RULES/render split in
step 2 shows up as a readable diff. The f-strings, the indentation and the prompt wording are
exactly as the notebook had them -- including the known defects, which later steps remove.
"""

def get_quality_grader_prompt(full_text):
  quality_grader_prompt = f"""
    You are an expert medical educator tasked with evaluating case reports for their diagnostic-reasoning value.
    You will be given full, uncleaned text that has just been extracted from PDF via OCR. You will have to understand the text because some parts are splited due to layout.
    These images are passed to you together with the case.
    The goal is to check if this case report can be use like a diagnostic teaching cases from medical textbooks.

    CASE REPORT EVALUATION RUBRIC
    >> HOW TO USE
    1. Read the entire case once without scoring.
    2. Re-read, taking notes.
    3. Inside <think>...</think>, write the reasoning that leads you to each score.
    4. Output only the five XML tags shown after the rubric—nothing else.

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
    | 5. STATED FINAL DIAGNOSIS (Yes / No)                  |
    | Is diagnosis clearly named? Yes → "Yes"; else → "No". |
    +-------------------------------------------------------------+

    Additional Rules:
    1. If the given article is not actually a case report, output ”NA” for all scores.
    2. Think through whether the final diagnosis can be reasonably deduced from the case presentation when determining the
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

    SUPPLIED CASE REPORT
    <full_text>
    {full_text}
    </full_text>
  """

  return quality_grader_prompt
