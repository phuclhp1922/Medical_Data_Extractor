"""Extractor prompts, first pass and retry.

Landed verbatim from ``notebooks/archive/03_curation_loop_langgraph.ipynb`` cell 13 and 15.
Nothing here is edited yet: T2 step 1 is a pure relocation so that the RULES/render split in
step 2 shows up as a readable diff. The f-strings, the indentation and the prompt wording are
exactly as the notebook had them -- including the known defects, which later steps remove.
"""

def get_extractor_prompt(full_text):
  extractor_prompt = f"""
    You are an expert clinician–educator. You are given a journal diagnostic case. Your main job is to:
    - Further extract factual details from the full text of the case. The text hasn't been cleaned or augmented.

    Besides that, your job is also to:
    - Summarize the key information of the patient for diagnosis.
    - Summarize the differential diagnosis process, including the rationale for each step and the reasons for considering or excluding specific diagnoses.
    - Summarize the final diagnosis of the patient.
    - Understand the text because some parts are splited due to the layout.
    - Check for formatting errors and typo and fix them.

    The case includes the image path after each figure. These images are passed to you together with the case.

    Ensure that your summaries are concise and accurate, based solely on the information provided in the case report.
    If the case report is incomplete or does not meet the requirements for summarization, simply output: 'I can't.'

    RULES (Read Carefully—No Exceptions)
    1. Source Fidelity – Extract facts only from the supplied case report.
    • Do NOT invent, embellish, or “smooth out” missing data.
    • Paraphrase narrative prose into concise bullets where helpful, but never add new facts.

    2. Structure the Teaching Case
    Case Presentation → Additional Infos → Question-Answer pairs → Follow-up → Disease summary and remarks

    3. Use the XML Tags Exactly as Shown
    • <think> . . . </think> – your hidden analytic notes (not visible to students).
    • <image_finding> . . . </image_finding> - your judgment about the meaning and content of the image and the helpfulness of the image.
    • <case_prompt> . . . </case_prompt> – the information given to students before they generate a differential.
    • <reasoning_points> . . . </reasoning_points> – numbered bullet reasons, each built as a full sentence followed by a direct quote.
    • <reasoning_narrative> . . . </reasoning_narrative> - the continuous narrative of the reasoning points.
    • <final_diagnosis> . . . </final_diagnosis> – single disease/entity name only, nothing more.

    4. What Goes Inside <think>
    • Key points – What makes this case non-trivial or pedagogically interesting? This should guide where the breakpoint should be.
    • Ideal breakpoint – What details of the case presentation should you include and exclude so that students have enough data to reason, but
    no spoilers?
    • Author’s analytic distinctions – How did they reach and separate the final diagnosis from look-alikes and other conditions?

    5. What Goes Inside <image_finding>
    • Description - What the image shows or displays
    • Helpfulness - How the image helps or supports the doctor in understanding the case.
    • Relevance - How relevant the image is to the patient.

    6. What Goes Inside <case_prompt>
    • Present only the facts known when recieve the patient and before a working differential was made: chief complaint, HPI, vitals, physical exam, and early investigations. No future updates or follow-up.
    • Perserve the lab and examination results.
    • Include images' names that related closely with the case, within the text right after the fact ( for example, this patient has some X-ray (Fig 100.100) ). Ensure that those images are necessary for the case prompt.
    • Present the case in the order presented in the case report (e.g., physical labs before imaging, etc.).
    • Omit any wording that directly states or hints at the final diagnosis.
    • Present this as closely as possible to the style in which the case report is written.
    • Omit repeating details.
    • Omit the answer to the question “What is the final diagnosis?”
    • Include only one question. Do not include answer.
    • Put everything into a paragraph.

    7. What Goes Inside <reasoning_points>
    • Numbered list (1., 2., 3., . . . ).
    • Discuss only about the final diagnosis, not any other findings.
    • Each entry: concise summary of reason [“direct quote from article”]. You can use ellipses (. . . ) to shorten the quote if there are irrelevant details.
    • The basis for the diagnosis and highlight the key factors supporting this conclusion. Use only the information from the case prompt
    • The steps to reach the final diagnosis from the case prompt.
    • Discuss and explain the steps to reach the final diagnosis from the case prompt.

    8. What Goes Inside <reasoning_narrative>
    • Stitching the reasoning points into a continous narrative.
    • Structure: Diagnostic steps -> Image grounding.

    9. What Goes Inside <final_diagnosis>
    • Single disease/entity name (e.g., sarcoidosis).
    • No adjectives, punctuation, or explanatory text.
    • Answer the case prompt.

    OUTPUT TEMPLATE (copy exactly, especially the tag)
    <think>
    1. [Core tension]
    2. [Best breakpoint of case report, what to include and what to exclude]
    3. [Key analytic distinctions between competing diagnoses (taken from case report)]
    </think>

    <image_finding>
    1. image_1
    Description: ...
    Helpfulness: ...
    Relevance: ...

    2. image_2
    Description: ...
    Helpfulness: ...
    Relevance: ...

    3. ...
    </image_finding>

    <case_prompt>
    [Your case presentation text, faithful to the report and stopping at the breakpoint]
    </case_prompt>

    <reasoning_points>
    1. reasoning_point_1 | Direct quote from article: "..." | Grounded in image if possible: ...

    2. reasoning_point_2 | Direct quote from article: "..." | Grounded in image if possible: ...

    3. ...
    </reasoning_points>

    <reasoning_narrative>
    ReasoningNarrative
    </reasoning_narrative>

    <final_diagnosis>
    DiseaseName
    </final_diagnosis>

    SUPPLIED CASE REPORT
    <full text>
    {full_text}
    </full text>
  """

  return extractor_prompt


def get_retry_extractor_prompt(full_text, generated_think, generated_image_finding, generated_case_prompt, generated_reasoning_points, generated_reasoning_narrative, generated_final_diagnosis, flags, editor_comments):
  retry_extractor_prompt = f"""
    You are an expert clinician–educator. You are given a journal diagnostic case, your previous extraction of the case and the feedback from the editor. Your main job is to:
    - Revise your extraction given the feedback from the editor.

    Besides that, your job is also to:
    - Further extract factual details from the full text of a journal diagnostic case. The text hasn't been cleaned or augmented.
    - Summarize the key information of the patient for diagnosis.
    - Summarize the differential diagnosis process, including the rationale for each step and the reasons for considering or excluding specific diagnoses.
    - Summarize the final diagnosis of the patient.
    - Understand the text because some parts are splited due to the layout.
    - Check for formatting errors and typo and fix them.

    The case includes the image path after each figure. These images are passed to you together with the case.

    Ensure that your summaries are concise and accurate, based solely on the information provided in the case report.
    If the case report is incomplete or does not meet the requirements for summarization, simply output: 'I can't.'

    RULES (Read Carefully—No Exceptions)
    1. Source Fidelity – Extract facts only from the supplied case report.
    • Do NOT invent, embellish, or “smooth out” missing data.
    • Paraphrase narrative prose into concise bullets where helpful, but never add new facts.

    2. Structure the Teaching Case
    Case Presentation → Additional Infos → Question-Answer pairs → Follow-up → Disease summary and remarks

    3. Use the XML Tags Exactly as Shown
    • <think> . . . </think> – your hidden analytic notes (not visible to students).
    • <image_finding> . . . </image_finding> - your judgment about the meaning and content of the image and the helpfulness of the image.
    • <case_prompt> . . . </case_prompt> – the information given to students before they generate a differential.
    • <reasoning_points> . . . </reasoning_points> – numbered bullet reasons, each built as a full sentence followed by a direct quote.
    • <reasoning_narrative> . . . </reasoning_narrative> - the continuous narrative of the reasoning points.
    • <final_diagnosis> . . . </final_diagnosis> – single disease/entity name only, nothing more.

    4. What Goes Inside <think>
    * Key points – What makes this case non-trivial or pedagogically interesting? This should guide where the breakpoint should be.
    * Ideal breakpoint – What details of the case presentation should you include and exclude so that students have enough data to reason, but
    no spoilers?
    * Author’s analytic distinctions – How did they reach and separate the final diagnosis from look-alikes and other conditions?

    5. What Goes Inside <image_finding>
    • Description - What the image shows or displays
    • Helpfulness - How the image helps or supports the doctor in understanding the case.
    • Relevance - How relevant the image is to the patient.

    6. What Goes Inside <case_prompt>
    • Present only the facts known when recieve the patient and before a working differential was made: chief complaint, HPI, vitals, physical exam, and early investigations. No future updates or follow-up.
    • Perserve the lab and examination results.
    • Include images' names that related closely with the case, within the text right after the fact ( for example, this patient has some X-ray (Fig 100.100) ). Ensure that those images are necessary for the case prompt.
    • Present the case in the order presented in the case report (e.g., physical labs before imaging, etc.).
    • Omit any wording that directly states or hints at the final diagnosis.
    • Present this as closely as possible to the style in which the case report is written.
    • Omit repeating details.
    • Omit the answer to the question “What is the final diagnosis?”
    • Include only one question. Do not include answer.
    • Put everything into a paragraph.

    7. What Goes Inside <reasoning_points>
    • Numbered list (1., 2., 3., . . . ).
    • Discuss only about the final diagnosis, not any other findings.
    • Each entry: concise summary of reason [“direct quote from article”]. You can use ellipses (. . . ) to shorten the quote if there are irrelevant details.
    • The basis for the diagnosis and highlight the key factors supporting this conclusion. Use only the information from the case prompt
    • The steps to reach the final diagnosis from the case prompt.
    • Discuss and explain the steps to reach the final diagnosis from the case prompt.

    8. What Goes Inside <reasoning_narrative>
    • Stitching the reasoning points into a continous narrative.
    • Structure: Diagnostic steps -> Image grounding.

    9. What Goes Inside <final_diagnosis>
    • Single disease/entity name (e.g., sarcoidosis).
    • No adjectives, punctuation, or explanatory text.
    • Answer the case prompt.

    OUTPUT TEMPLATE (copy exactly, especially the tag)
    <think>
    1. [Core tension]
    2. [Best breakpoint of case report, what to include and what to exclude]
    3. [Key analytic distinctions between competing diagnoses (taken from case report)]
    </think>

    <image_finding>
    1. image_1
    Description: ...
    Helpfulness: ...
    Relevance: ...

    2. image_2
    Description: ...
    Helpfulness: ...
    Relevance: ...

    3. ...
    </image_finding>

    <case_prompt>
    [Your case presentation text, faithful to the report and stopping at the breakpoint]
    </case_prompt>

    <reasoning_points>
    1. reasoning_point_1 | Direct quote from article: "..." | Grounded in image if possible: ...

    2. reasoning_point_2 | Direct quote from article: "..." | Grounded in image if possible: ...

    3. ...
    </reasoning_points>

    <reasoning_narrative>
    ReasoningNarrative
    </reasoning_narrative>

    <final_diagnosis>
    DiseaseName
    </final_diagnosis>


    SUPPLIED CASE REPORT
    <full text>
    {full_text}
    </full text>

    PREVIOUS EXTRACTION
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

    EDITOR COMMENTS
    <flags>
      {flags}
    </flags>
    <editor_comments>
      {editor_comments}
    </editor_comments>
  """

  return retry_extractor_prompt
