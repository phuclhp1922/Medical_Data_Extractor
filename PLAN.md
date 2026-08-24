# Medical VLM Data Curator — Rebuild, Calibration & Study Plan

> Working plan for turning `LangGraph_Multimodal_Extraction.ipynb` into a CV-grade project.
> Budget: ~10 hrs/week × 4 weeks ≈ 40 hours.

---

## 1. Where the project actually stands

**What's built and working.** A real LangGraph state machine (`StateGraph`, conditional
edges — not plain LangChain) that turns MinerU-extracted case reports into teaching cases:

```
load_case → grade_case →(pass?)→ extract_case → review_case →(flags?)→ extract_case ⟲
                       ↘ END                                          ↘ END
```

Source corpus: 93 cases from *Clinical Cases in Tropical Medicine* (Elsevier, 2022),
extracted with **MinerU 2.5.4**. All 93 `_origin.pdf` files are still on disk at
`d:/extracted_data/*/auto/`, so the extraction stage is fully reproducible.

**What blocks it from being a portfolio piece:**

1. Five correctness bugs — one of which silently emptied your multimodal dataset.
2. The verifier has never been validated. *"An LLM judged it"* is not a claim.
3. It's a notebook, so it reads as coursework rather than engineering.

**Two upgrade items from your original list are already done** — don't spend time on them:
- It's *already* an explicit LangGraph state machine with named nodes.
- The *grader* already has a proper graded rubric (the *editor* does not).

Also: `ChatGoogleGenerativeAI` is instantiated at cell 18 and never called. Every request
goes through the raw `genai.Client`. LangChain is dead weight — drop it, keep LangGraph.

---

## 2. Verified bugs

Every number below I measured against the real corpus, not estimated.

### 2.1 The figure regex matches nothing — your multimodal data has no images

```python
r'(Fig\.\s*\d+\.?\d*)\s*!\[\]\((images/[a-f0-9]+\.jpg)\)'
```

This expects the figure label **before** the image. MinerU emits the image **first**, with
the caption after and the label frequently mangled mid-caption by OCR:

```
![](images/7274b77….jpg)
�   Oral bleeding in Ebola virus disease. (Bausch, D.G., 2008. Viral Fig. 1.1hemorrhagic fevers…
```

**Measured: 0 of 93 cases match. 0 of 142 figures resolved.**

`extract_image_paths` is never called by the graph, so that alone is harmless — but
**cell 52 uses the same regex** to build `benchmark_image/`. Its
`if image_path_pairs and figures:` branch can never fire, so the directory is empty and
`benchmark_image.zip` shipped with no images. This is the single highest-impact fix.

**Secondary bug in the same area:** `load_images_as_base64` globs all **219** `.jpg` files,
but only **142** are real figures. You are sending ~77 tables/decorations to the model as
unordered, unlabelled images — while the prompt asks it to cite "(Fig 100.100)". It cannot.

### 2.2 The editor audits against an empty rubric

`review_case` calls `get_editor_prompt("", …)`. The prompt says *"Here is the original
guideline the draft was produced in accordance with: {extractor_prompt}"* and receives an
empty string. Your verifier has been checking compliance against nothing.

⚠️ **Don't just pass `get_extractor_prompt(full_text)`** — that embeds the case text a
second time and inflates cost. Split each prompt into `RULES` (constant) + `render(inputs)`
and pass `RULES` alone.

### 2.3 No retry cap

`loop_count` is initialised in `load_case`, incremented in `review_case`, and **never read
by anything**. `review_route` only checks the flags. The loop is bounded solely by
LangGraph's default `recursion_limit=25` (~11 refines), which raises `GraphRecursionError`,
gets swallowed by the bare `except Exception` in cell 34, and **silently drops the case** —
losing exactly the hard cases you most want to study.

> **Reference implementation** — `langchain-ai/data-enrichment`, `src/enrichment_agent/graph.py`.
> Its `route_after_checker` is exactly this fix:
> ```python
> if state.loop_step < configurable.max_loops:
>     return "call_agent_model"
> return "__end__"
> ```
> with `loop_step` incremented in the agent node. Note it keeps `max_loops` in
> `configuration.py` (runtime config) rather than as a module constant — copy that too.

### 2.4 `"NA"` crashes the grader

The grader prompt says *"If the given article is not actually a case report, output 'NA'
for all scores."* Then `grade_case` does `int(grading["case_presentation_score"])` →
`ValueError` → swallowed → case dropped. The non-case-reports you deliberately asked it to
flag are the ones that crash.

### 2.5 Contradictory flag vocabulary (breaks the error taxonomy)

The editor prompt defines the controlled vocabulary as
`CASE PROMPT HALLUCINATION / FINAL DIAGNOSIS IN CASE PROMPT / INSUFFICIENT INFO FOR
DIAGNOSIS / DIAGNOSTIC REASONING HALLUCINATION / OTHER` — but the worked example
immediately below uses `SOURCE_FIDELITY` and `REASONING_EXTRA_INFO`, which aren't in that
list. **Models follow the example.** Same problem with the tag itself: `<editor comments>`
(spec) vs `<editor_comments>` (example, and what the parser looks for).

Until these agree, the flags are inconsistent and the error taxonomy is unusable.

> **Reference implementation** — the same template's reflection node calls
> `with_structured_output(InfoIsSatisfactory)`, returning a typed object with
> `reasoning / satisfactory / improvement_instructions`. That is structurally your
> `flags / editor_comments`, but schema-enforced.
>
> **This kills §2.5 and §2.6 together.** An enum field cannot drift between the spec and
> the worked example the way your prose vocabulary did, and there is no tag to fail to
> parse. Prefer this over fixing the prompt text by hand — a schema makes the bug
> *impossible* rather than merely *fixed*.

### 2.6 Also worth fixing

- `split_text_to_tag` KeyErrors whenever the model omits a tag → case dropped.
- `select_synthesized_fields` saves only 6 fields, **discarding the grading scores and
  `image_finding`**. You cannot audit past runs or calibrate without regenerating everything.
- `data.dropna()` (cell 42) silently discards incomplete rows, so you have no honest yield.
- `CONFIG` is `temperature=0.1` globally — self-consistency sampling does nothing as-is.

---

## 3. The figure resolver design (validated)

I tested the replacement approach against the corpus. Results:

| Check | Result |
|---|---|
| Caption regex `Fig\.?\s*(\d+)[.\-](\d+)` finds a label | 117 / 142 (82%) |
| Caption chapter == leading number in directory name | 117 agree, **1 disagree**, 24 no label |
| Positional index (nth image in reading order) == caption figure number | **116 / 117** |

So two independent methods agree on 116 of 117 checkable figures. The design:

1. **Positional:** nth image in `_content_list.json` reading order → `Fig {chapter}.{n}`,
   where `chapter` is the leading number in the case directory name.
2. **Caption:** regex over `image_caption`, **accepted only if its chapter matches the
   directory chapter**.
3. Cross-check; report the disagreement rate as a QA metric.

**Why the chapter guard matters — a real example.** Case 22's only image has a caption
containing `Fig. 43.1` (a cross-reference to another chapter, not its own label), while the
body text refers to `Fig. 22.1`. Naive "first regex match wins" mislabels it. The chapter
guard rejects it and positional inference gets it right. *This is the entire argument for
using two methods, and it's a good thing to be able to narrate in an interview.*

One residual disagreement (case 24, positional says figure 1, caption says `24.12`) needs
manual inspection — likely a figure reproduced from elsewhere in the book.

Also feed the **caption text** alongside each image. Grounding improves a lot when the model
gets `Fig 1.1 — "Oral bleeding in Ebola virus disease"` instead of an anonymous JPEG.

---

## 4. Cost estimate

Measured: 93 cases, 848,634 markdown chars (~2,280 tokens/case avg, 3,220 max),
142 true figures across 219 image files.

| Workload | Input | Output |
|---|---|---|
| One case, single pass (grade + extract + review) | ~17k | ~3.4k |
| One case, with avg 1.5 refine cycles | ~23k | ~4.6k |
| **Full 93-case run** | **~2.1M** | **~0.43M** |
| **Judge-only calibration pass, 30 cases** | **~200k** | **~15k** |

At current mid-tier hosted rates a full run is roughly **$2–8**; the judge-only calibration
pass is **well under $1**. (Verify live pricing yourself — don't take my dollar figures on
faith; the token counts are the reliable part.)

**Conclusion: money is not your constraint — free-tier rate limits are.** That's what all
the `time.sleep(5)` calls are working around. Your $20–50 estimate was ~10× too pessimistic.

**Decision:** keep Gemma/Flash as the extractor; use a **different model family** as judge
on the calibration subset. Right now `gemma-3-27b-it` extracts, grades, *and* audits its own
output — textbook self-enhancement bias (see MT-Bench in the reading list).

---

## 5. Repo layout

```
medical-vlm-curator/
  pyproject.toml           # pin mineru==2.5.4
  .gitignore               # data/, outputs/, *.pdf, *.jpg  <- non-negotiable
  README.md
  DATASHEET.md             # dataset card, Gebru et al. format
  src/curator/
    config.py              # models, thresholds, retry cap, corpus path via env var
    figures.py             # NEW - deterministic figure resolver (§3)
    parsing.py             # tolerant replacement for split_text_to_tag
    prompts/               # grader.py extractor.py editor.py - RULES split from inputs
    graph.py  nodes.py     # LangGraph state machine
    extract.py             # MinerU wrapper
    run.py                 # CLI: curate / resume / export
  src/curator/eval/
    sample.py label_cli.py agreement.py taxonomy.py
  tests/
  notebooks/01_demo.ipynb  # thin driver, kept for narrative
  reports/                 # committed plots + metrics (safe to publish)
```

This layout is close to `langchain-ai/data-enrichment`'s `src/enrichment_agent/`
(`graph.py` / `state.py` / `configuration.py` / `prompts.py` / `utils.py`) — independent
confirmation that it's the idiomatic LangGraph structure rather than something invented
for this project. Two things worth copying wholesale:

- **`configuration.py`** — runtime config (model, `max_loops`, thresholds) as a typed
  object rather than module-level constants. Makes the calibration sweeps in Phase 2 much
  easier, since you can vary judge model and thresholds without editing code.
- **`tests/casettes/*.yaml`** — VCR cassettes recording real LLM responses so integration
  tests replay offline. **Given that rate limits are your binding constraint (§4), this is
  the single most useful thing in that repo.** Record once, then iterate on parsing,
  routing, and scoring logic for free.

**Data governance — do this before anything else.** The corpus is derivative of a
copyrighted Elsevier textbook. Never commit it. Point at it with a `CURATOR_CORPUS`
environment variable (it already lives *outside* the project at `d:/extracted_data`, which
is exactly right). Commit code, aggregate metrics, plots, and 2–3 *paraphrased* examples.
Explain the reasoning in `DATASHEET.md` — deliberate handling of licensed clinical data is
a genuine hiring signal, not an obstacle.

---

## 6. Execution phases

### Phase 0 — Foundation (week 1, ~8h)
1. `git init`, package skeleton, **`.gitignore` before any data goes near the repo**.
2. Port the notebook into `src/curator/` **verbatim, no behaviour changes**. Get it running
   on one case. This checkpoint proves the port is faithful before you start changing things.
3. Drop the unused LangChain import. Add `pytest` + `ruff` + one smoke test.
4. **Record VCR cassettes for 3 cases** (§5) before touching any logic. Everything in
   Phase 1 except the live-API paths then becomes testable offline and instantly —
   which is what makes a 10 hrs/week budget realistic against a rate-limited free tier.

### Phase 0.5 — Reproducible extraction (week 1, ~4h)
- `extract.py` wrapping MinerU, pinned to `2.5.4`.
- Re-run on 5 `_origin.pdf` files, diff against existing output.
- **If it diverges, do not regenerate the corpus.** Treat current `extracted_data/` as the
  golden reference and document the divergence. Regenerating invalidates any calibration.

### Phase 1 — Correctness fixes (week 2, ~10h)
Fix in this order; each is independently verifiable:
1. **Figure resolver** (§3) — highest impact, and testable with zero API calls.
2. Editor empty-rubric → `RULES` + `render()` split.
3. `MAX_REFINES = 3` with a **recorded** `refine_exhausted` terminal state.
4. `"NA"` → explicit `not_a_case_report` outcome.
5. Tolerant tag parser + Pydantic schema (this also delivers your "firm up the schema" item,
   keeping only `reasoning_points` / `reasoning_narrative` as free text).
6. Persist **full** state per case, not six fields.

### Phase 2 — Judge calibration (week 2–3, ~12h) ← the centrepiece
**Calibrate the grader first** — it already emits numeric scores, so it yields a real
statistic immediately. The editor must be fixed first, or you'd just be measuring the
empty-rubric bug.

1. Sample 30 cases stratified across pass/borderline/fail, fixed seed, **recorded**.
2. **Label them yourself, blind**, before seeing model scores (~4h). Blindness is what makes
   the number credible — enforce it in the CLI, don't just intend it.
3. **Get a second rater** for even 15 of the 30. Without a human–human ceiling you cannot
   distinguish "the judge is bad" from "the task is ambiguous" — and that is the first thing
   a sharp interviewer will ask.
4. Compute: **quadratic-weighted Cohen's κ** for the 1–5 criteria, plain κ for the Yes/No
   ones, Spearman ρ on totals, a per-criterion breakdown, and a **confusion matrix on the
   pass/fail gate** (which is what actually matters). Expect `transparency` and
   `images_usefulness` to be noisiest — that's a finding, not a failure.
5. Re-run with an **independent judge model**, report both numbers. Costs < $1.
6. **Report the funnel honestly:** N PDFs → passed grading → extracted → passed editor →
   final. That funnel *is* the result. "93 cases" in a filename is not a yield.

### Phase 3 — Rubric, taxonomy, self-consistency (week 4, ~8h)
1. Fix the flag vocabulary (§2.5) — nothing downstream works until this is consistent.
2. Give the editor a **graded** rubric, not binary flags, so it becomes calibratable too.
3. Error taxonomy over collected flags; report the distribution.
4. Self-consistency: sample the judge k=5 at temperature ~0.7. Check whether high-variance
   cases correlate with your human disagreements — if they do, you have a cheap ambiguity
   detector, which is a genuinely strong result.

### Phase 4 — Write-up (~4h)
`README.md` with the architecture diagram (`app.get_graph().draw_mermaid_png()` already
produces it), the funnel table, the calibration table, and the taxonomy chart.
`DATASHEET.md` for provenance. Short and metric-forward.

**If you run out of time:** the image fix and the calibration number are the two items that
carry the CV claim. Everything after Phase 2 is upside.

---

## 7. What to read and learn

Sequenced so each item lands just before you need it. The "be able to" column is the point —
reading without being able to do the thing doesn't survive an interview.

### Core four — these carry most of the value

| When | Read | Be able to… |
|---|---|---|
| **Before Phase 2** | **"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"** — Zheng et al., NeurIPS 2023 | Name the three judge biases (position, verbosity, **self-enhancement**) and explain which one your current setup has and how you removed it. Read this one first. |
| **Before Phase 2** | **"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"** — Liu et al., EMNLP 2023 | Explain chain-of-thought + form-filling rubric scoring, and how they correlate judge output with human ratings. Closest published analogue to what you're building. |
| **Phase 2** | **Inter-rater reliability**: Cohen's κ, quadratic-weighted κ, Krippendorff's α. Any solid primer + `sklearn.metrics.cohen_kappa_score(weights="quadratic")` | Justify *why weighted κ for ordinal scales* and *why raw agreement % is misleading*. You must be able to defend the statistic, not just print it. |
| **Phase 1** | **LLaVA-Med** — Li et al., 2023 | Recognise that your figure resolver is the same figure–caption grounding recipe, and cite it in your README. |

### Supporting, by phase

**Phase 0/1 — engineering craft**
- **`langchain-ai/data-enrichment`** (MIT) — read the *source*, not the README. It is the
  closest well-built analogue to your graph: extract → reflect → retry with an enforced
  loop bound. Read in this order: `state.py` → `graph.py` → `configuration.py`. Referenced
  concretely in §2.3, §2.5, and §5.
  *Note:* the repo your friend shared (`fuisl/data-extraction`) is an unmodified copy of
  this template — one commit, no changes — so read the upstream original instead. Credit it
  in your README; MIT requires attribution.
- LangGraph docs: *persistence & checkpointers*, *recursion limit*, conditional edges.
  Checkpointing gives you resume-on-crash for free and fixes your retry cap properly.
- Pydantic v2 structured output; **Instructor** library, or Gemini's native
  response-schema mode. Either removes your regex tag parsing entirely.
- **VCR / `pytest-recording`** for cassette-based LLM tests (§5). Learn this early —
  it decouples your iteration speed from the free-tier rate limit.
- Packaging + tooling: `uv`, `ruff`, `pytest`, `typer`. Cheap to learn, immediately visible
  in a repo.

**Phase 2 — medical evaluation rubrics**
- **Med-PaLM 2 — "Towards Expert-Level Medical Question Answering with Large Language
  Models"** — Singhal et al., 2023. Its physician-rater rubric is the best available
  template for phrasing your criteria in clinically legible terms.

**Phase 3 — data curation philosophy**
- *"Textbooks Are All You Need"* (phi-1) and the **FineWeb** technical report. Both argue
  quality filtering over raw volume — exactly your grader's thesis. Strong interview material.

**Phase 4 — documentation**
- **"Datasheets for Datasets"** — Gebru et al. Follow the format directly. Cheap to write,
  disproportionately impressive to reviewers.

**Optional breadth**
- Any 2024–25 *"Survey on LLM-as-a-Judge"*.
- **MedXpertQA**, **PMC-VQA**, **PathVQA** — for how medical VQA benchmarks report curation.

> These are titles + authors, not links. Search for them — don't trust a URL I gave you
> from memory.

### Rehearse these questions

They are the actual point of the exercise:

- *Why weighted κ and not accuracy?*
- *How do you know the judge isn't just agreeing with itself?*
- *What's your inter-annotator ceiling?*
- *What did the pipeline reject, and was it right to?*
- *Your judge disagreed with you most on "transparency". Why, and what did you do?*

---

## 8. Verification

- **Phase 0.5:** MinerU re-run on 5 PDFs diffs clean against the golden copy.
- **Phase 1:** `tests/test_figures.py` asserts ≥117/142 figures resolved and **0 unlabelled
  images sent to the model**; snapshot test on one case's figure map. Run the graph on 3
  cases and confirm `benchmark_image/` is **non-empty** — the headline regression fix.
- **Phase 1:** unit tests for `"NA"` grading, missing-tag parsing, and `MAX_REFINES=3`
  producing a *recorded* outcome rather than an exception.
- **Phase 2:** the labelling CLI must be *unable* to show model scores before a human label
  is committed — enforced in code, and worth a test.
- **End to end:** a full run emits the funnel table, and every input case lands in exactly
  one terminal state. **No silent drops.**

## 9. Risks

- **MinerU may not reproduce byte-identically** even pinned (model weights, CUDA). Golden
  reference stays authoritative; the extraction stage is for documentation, not regeneration.
- **Single-annotator calibration is weak.** Get a second rater for even half the subset. If
  genuinely impossible, state the limitation in the README — saying it plainly is stronger
  than quietly having it.
- **Rate limits, not cost, dominate wall-clock.** Build resume-from-checkpoint early (you
  have a crude version already in `if out_file.exists()`).
- **Scope creep.** Phases 3–4 are optional. Protect Phases 1–2.
