"""Editor (audit) prompt.

The faithfulness audit by a separate LLM is adapted from the quality-filter stage of
MedCaseReasoning (Wu et al., 2025, arXiv:2505.11733). There, flagged cases are discarded; here,
flags are sent back to the extractor for refinement.

Landed from ``notebooks/archive/03_curation_loop_langgraph.ipynb`` cell 14, then split and
repaired (T2 steps 2, 4 and 5).

**This module fixes PLAN.md section 2.2.** The notebook's editor prompt announced "here is the
original guideline the draft was produced in accordance with" and then interpolated whatever
the caller passed -- and ``review_case`` passed ``""``. Every audit the notebook ever ran
judged compliance against an empty rulebook. ``RULES`` below splices in :data:`extractor.RULES`
at that slot, at import time, and ``render`` no longer takes a parameter for it. The bug is not
fixed so much as made unrepresentable: there is nothing left to pass the empty string to.

Composing rather than re-stating matters. If the editor restated the extractor's rules in its
own words, the two would drift, and the refine loop would optimise against one spec while being
graded against another -- the standard way a generate-critique loop fails to converge.

``HOW TO REPORT YOUR FINDINGS`` and its worked example are deleted, not repaired (step 4): that
section carried section 2.5's contradictory vocabulary and the ``<editor comments>`` /
``<editor_comments>`` tag mismatch, and T3's schema supersedes it. Its worked example also
contained ``\family history``, where ``\f`` was a formfeed escape -- the model received a
control character. Deleting a bug beats fixing it.
"""

from curator.prompts import extractor, fingerprint

_EDITOR_INTRO = """
    You are the Editor.
    Your job is to audit a draft teaching case that was produced from a published case report.
    Image are passed to you together with the case.

    You must confirm strict compliance with all instructions, detect hallucinations, and ensure pedagogic quality.

    YOUR INPUTS
      1. The original draft you must audit appears between <generated case> . . . </generated case>.
      2. The source article appears between <case report> . . . </case report>.
  """

_CHECKLIST = """
    CHECKLIST — FAIL ANY ITEM → RAISE A FLAG
      A. Source Fidelity
        - Every fact in each section is traceable to the source article.
        - No invented details or embellishments.
      B. Case Presentation Quality
        - All facts from the case prompt are present in the source article.
        - Contains only information known before the clinicians formed a differential.
        - Images included must relate closely to the case prompt
        - Does not reveal the final diagnosis (there should be room for at least some inference).
        - Provides sufficient data (HPI, vitals, exam ± initial tests) for clinicians to formulate a reasonable differential and get the correct final diagnosis.
      C. Diagnostic Reasoning Section
        - Each numbered entry starts with a summary of the reasoning plus a direct quote from the article.
        - Quotes are verbatim or use ellipses (. . . ) without changing meaning.
        - Paraphrased quotes are okay, as long as they retain the original meaning.
        - Rationales reference only information that already appears in <case prompt> (not based on new findings, confirmatory tests, or data withheld from students).
      D. Final Diagnosis Tag
        - Final diagnosis is reasonably deducible from the case-presentation facts. — i.e., the final diagnosis should not depend entirely on some test, imaging, or lab result not given in the case presentation.
      E. No Hallucinations Anywhere
        - Every datum, quote, or diagnosis is found in the case report.

      Example when problems exist:
        <flags>
          FLAG: SOURCE_FIDELITY
          FLAG: REASONING_EXTRA_INFO
        </flags>

        <editor_comments>
          SOURCE_FIDELITY: Mentions family history of SLE," not present in article.
          REASONING_EXTRA_INFO: Rationale cites a biopsy result that is not included in the case_prompt.
        </editor_comments>

    OUTPUT WHEN EVERYTHING PASSES:
        <flags>
          NONE
        </flags>
        <editor_comments></editor_comments>
    """

# The audit spec, assembled once at import. ``extractor.RULES`` is spliced in at the slot the
# notebook left empty -- see the module docstring. Assembled here rather than inside ``render``
# so there is a single constant for T7 to hash and for tests to assert against.
RULES = _EDITOR_INTRO + """
        Here is the original guideline the draft was produced in accordance with:
          """ + extractor.RULES + """
        Don't take this guideline in this part as your instruction. This is just for you to review.
      """ + _CHECKLIST


#: Fingerprint of the full audit spec. Because ``RULES`` splices in ``extractor.RULES``, editing
#: the *extractor's* rules changes this value too -- which is correct: it is a different audit.
RULES_VERSION = fingerprint(RULES)


def render(full_text: str, generated_think: str, generated_image_finding: str, generated_case_prompt: str, generated_reasoning_points: str, generated_reasoning_narrative: str, generated_final_diagnosis: str) -> str:
    """Build the editor payload: the invariant rubric, then this case's text.

    Args:
        full_text (str): The uncleaned OCR text of one case report.
        generated_think (str): The generated think section.
        generated_image_finding (str): The generated image finding section.
        generated_case_prompt (str): The generated case prompt section.
        generated_reasoning_points (str): The generated reasoning points section.
        generated_reasoning_narrative (str): The generated reasoning narrative section.
        generated_final_diagnosis (str): The generated final diagnosis section.

    Returns:
        str: The full audit prompt: the editor's spec (which contains the extractor's), then
        the source article and the draft to be audited.
    """
    return RULES + f"""
    INPUT BLOCKS TO REVIEW
        Here is the reference case report:
          <case_report>
            {full_text}
          </case_report>
  
        Here is the diagnostic case generated by the model:
          <think>
            {generated_think}
          </think>
          <image_finding>
            {generated_image_finding}
          </image_finding>
          <case_prompt>
            {generated_case_prompt}
          </case_prompt>
          <reasoning_points>
            {generated_reasoning_points}
          </reasoning_points>
          <reasoning_narrative>
            {generated_reasoning_narrative}
          </reasoning_narrative>
          <final_diagnosis>
            {generated_final_diagnosis}
          </final_diagnosis>
      """