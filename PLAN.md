# Medical VLM Data Curator — Rebuild, Calibration & Study Plan

> Working plan for turning `03_curation_loop_langgraph.ipynb` into a CV-grade project.
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
extracted with **MinerU 2.5.4** (`_backend: pipeline`, confirmed identical across all 93
`_middle.json`). All 93 `_origin.pdf` files are still on disk at `d:/extracted_data/*/auto/`.

### 1.1 The pipeline has three stages, in three notebooks

`notebooks/archive/` holds the whole lineage. Read them in this order — the plan below
refers to all three:

| # | Notebook | What it does |
|---|---|---|
| 1 | `01_pdf_extraction_mineru.ipynb` | PDF → markdown + images. Six lines of Colab shell. |
| 2 | `02_curation_loop_python.ipynb` | First grade → extract → edit loop, plain Python `while`. |
| 3 | `03_curation_loop_langgraph.ipynb` | The same loop rebuilt as a `StateGraph`. |

**The critical fact these reveal: there is a fourth, missing stage.** Notebooks 2 and 3 both
consume `extracted_case_report_image_filtered.zip` from Google Drive — a corpus whose
directories are named `case_1 … case_93`, containing **only** `{case}.md` and `images/`.
That is *not* what MinerU produced. Comparing it to `d:/extracted_data`:

| | `d:/extracted_data` (raw MinerU) | `extracted_case_report_image_filtered` (what ran) |
|---|---|---|
| Directory names | `28---A-67-Year-Old-Female-Expatriate…` | `case_28` |
| Files present | `.md`, `_content_list.json`, `_middle.json`, `_model.json`, `_origin.pdf`, `_layout.pdf`, `_span.pdf`, `images/` | `.md`, `images/` |
| Figure markup | `![](images/hash.jpg)` then caption after | `Fig. 28.1 ![](images/hash.jpg)` |

The image hashes are **identical between the two** (spot-checked cases 5, 28, 33, 43 — all
present), so this is the same MinerU run, post-processed: directories renamed, auxiliary
files deleted, and **the markdown rewritten so figure labels precede their images**.

**The script that did that is not in any of the three notebooks and is not on disk.** Your
pipeline's actual input therefore cannot currently be regenerated from your raw corpus. This
is the single biggest reproducibility hole in the project, it was invisible until now, and
Phase 0 must close it. The good news: `figures.py` (§3) *is* that missing stage, done
properly and from `_content_list.json` rather than from edited markdown.

**What blocks it from being a portfolio piece:**

1. **The verifier has never rejected a single case** — 0 flags in 51 (§2.6). *"An LLM
   judged it"* is not a claim, and right now it isn't even a filter.
2. **A missing pipeline stage** (§1.1): the corpus the pipeline consumes was produced by a
   script that no longer exists, so the input cannot be rebuilt.
3. Six correctness bugs, mostly silent-failure ones (§2).
4. It's a notebook, so it reads as coursework rather than engineering.

**Two upgrade items from your original list are already done** — don't spend time on them:
- It's *already* an explicit LangGraph state machine with named nodes.
- The *grader* already has a proper graded rubric (the *editor* does not).

Also: `ChatGoogleGenerativeAI` is instantiated at cell 18 and never called. Every request
goes through the raw `genai.Client`. LangChain is dead weight — drop it, keep LangGraph.

---

## 2. Verified bugs

Every number below I measured against the real corpus, not estimated.

### 2.1 The figure regex — corrected, and the real story is more interesting

> **Correction.** An earlier draft of this plan said the regex matched nothing and that
> `benchmark_image/` therefore shipped empty. **That was wrong.** I had measured against
> `d:/extracted_data`, but the pipeline never ran on `d:/extracted_data` — it ran on the
> post-processed corpus described in §1.1, where the markup had already been fixed. The
> saved outputs in notebook 3 prove it: `benchmark_image/` shipped **70 `.jpg` files across
> 56 case directories**, none of them empty. The image pipeline worked.

```python
r'(Fig\.\s*\d+\.?\d*)\s*!\[\]\((images/[a-f0-9]+\.jpg)\)'
```

The regex expects the label **before** the image. That is true of the corpus that ran, and
false of raw MinerU output, which emits the image first with the caption after and the label
frequently mangled mid-caption by OCR:

```
![](images/7274b77….jpg)
�   Oral bleeding in Ebola virus disease. (Bausch, D.G., 2008. Viral Fig. 1.1hemorrhagic fevers…
```

**Measured on raw MinerU output: 0 of 93 cases match, 0 of 142 figures resolved.**
**Measured on the corpus that actually ran: 80 of 93 cases matched, 158 images uploaded.**

So the regex is not a live bug — it is a **hidden dependency on an unreproducible manual
transform**. It works only as long as you use a zip file whose generating script no longer
exists. The moment you re-extract anything, it silently returns to matching zero. That is a
worse failure mode than a plain bug, because it fails silently and only under change.

**Where images were actually lost.** The 93 → 56 attrition is real, and it is not the regex:

| Stage | Cases | Where it happens |
|---|---|---|
| Cases in corpus | 93 | |
| Rows surviving `data.dropna()` | 78 | cell 42 — **15 dropped silently** |
| Whose `case_prompt` cites any `Fig. N.M` | 56 | cell 52 — 22 cases cite no figure |
| Case dirs written to `benchmark_image/` | 56 (70 jpgs) | |

The 22 are the substantive finding: the extractor was *told* to cite figures in the case
prompt ("Include images' names (for example, Fig 100.100)"), and in 22 of 78 cases it simply
didn't — so a figure that exists, and was uploaded to the model, never reaches the dataset.
That is a prompt-compliance failure worth measuring, not a regex bug.

**Secondary bug, still real:** `load_images_as_base64` globs all **219** `.jpg` files, but
only **142** are real figures. You are sending ~77 tables/decorations to the model as
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
> **This kills §2.5 and the tag-parsing bug in §2.7 together.** An enum field cannot drift between the spec and
> the worked example the way your prose vocabulary did, and there is no tag to fail to
> parse. Prefer this over fixing the prompt text by hand — a schema makes the bug
> *impossible* rather than merely *fixed*.

### 2.6 The editor has never rejected anything — 0 flags in 51 cases

This is the most important number in this document and it comes straight out of notebook 2's
saved output. Cell 23 prints `{source} extracted with {count} loops` for every case. All
**51** lines read `with 1 loops`. `count` starts at 1 and increments once per refine, so
**every single case passed the editor on the first attempt.**

An auditor with a 0% rejection rate across 51 samples is not auditing. There are **two
candidate causes, and they are confounded** — separating them is now a Phase 2 deliverable:

1. **The empty rubric (§2.2).** It was auditing compliance against an empty string.
2. **The model was forced down by quota.** `gemma-3-27b-it` was not a design choice. Google
   cut the free-tier daily limit on Gemini and squeezed general free capacity, so the run
   switched to Gemma. The code still carries the fossils: the variable is named **`MODEL_2`
   with no `MODEL_1` anywhere in any notebook**, and the key is **`GEMINI_API_KEY_3`** — the
   third of a rotating set. Both notebooks inherited the constant unchanged.

Cause 2 matters more than it looks. A 27B instruction-tuned model asked to perform adversarial
audit against a long checklist is close to worst-case for that model class: critique is harder
than generation, and small models default to agreement. Rubber-stamping is the expected
behaviour, not an anomaly. **So do not assume fixing the rubric alone will fix the editor** —
and equally, do not write the editor off as a bad idea. It has never been tested under
conditions where it could plausibly work.

The consequences of the 0% rate, either way:

- The refine loop **has never executed in production**, in either notebook.
- Which means notebook 2's retry loop has a latent crash nobody has ever hit:
  ```python
  # 8 parameters expected, 7 passed — generated_image_finding is missing
  editor_prompt = get_editor_prompt(retry_extractor_prompt, full_text, generated_think,
                                    generated_case_prompt, generated_reasoning_points,
                                    generated_reasoning_narrative, generated_final_diagnosis)
  ```
  `TypeError` on the first refine, in a cell with **no `try`/`except`** — it would have
  aborted the entire run. The 0% flag rate is the only reason that run ever finished.
- The same loop is `while flags != "NONE"` with **no cap at all**. §2.3's missing retry cap
  is inherited, not introduced.
- Notebook 2's retry also calls the editor with `contents=editor_prompt` (no images), while
  the first pass sends `[editor_prompt, *images]`. The refine path is blind.

**What this does to the plan:** it turns Phase 2 from "nice to have a number" into the load-
bearing item. You cannot report a yield of "51 of 93 cases passed editorial review" when the
editor passes everything. Until it discriminates, it is a no-op in the pipeline, and the
first calibration run may well produce κ ≈ 0 — **which is a publishable finding for this
project, not a failure.** Getting an honest zero and explaining it is a stronger interview
story than a suspicious 0.8.

**And the confound is the opportunity.** "The judge rubber-stamped everything" is a bug
report. "The judge rubber-stamped everything; I isolated how much was the prompt and how much
was the model I'd been forced onto by a quota change, and here are both numbers" is an
engineering result. The second one is nearly free — see Phase 2, step 5.

### 2.7 Also worth fixing

- `split_text_to_tag` KeyErrors whenever the model omits a tag → case dropped.
- `select_synthesized_fields` saves only 6 fields, **discarding the grading scores and
  `image_finding`**. You cannot audit past runs or calibrate without regenerating everything.
- `data.dropna()` (cell 42) silently discards incomplete rows, so you have no honest yield.
- `CONFIG` is `temperature=0.1` globally — self-consistency sampling does nothing as-is.

---

## 3. The figure resolver design (now validated against real ground truth)

**New: you have a labelled test set and didn't know it.** Notebook 3's cell 52 printed every
`(Fig. N.M, images/hash.jpg)` pair it resolved. Those outputs survive in the pre-strip backup
of the notebook, and they are recoverable: **118 figure→file pairs across 67 cases, and all
118 `.jpg` files verified present in `d:/extracted_data`.** That is a genuine ground truth
produced by a different method (the hand-fixed markdown), so testing the new resolver against
it is not circular for the caption path.

Scored against those 118 pairs:

| Method | Correct |
|---|---|
| **Positional** — nth image in `_content_list.json` order → `Fig {chapter}.{n}` | **118 / 118 (100%)** |
| **Caption regex** `Fig\.?\s*(\d+)[.\-](\d+)` + chapter guard | 96 / 118 (81%) |
| Either method correct | 118 / 118 |
| Both fire **and** agree | 96 — **zero disagreements** |

Corpus-wide (all 142 figures, no ground truth available): caption regex finds a label on
117/142 (82%), the chapter guard rejects 1, and 24 have no parseable label at all.

**This inverts the design in the earlier draft.** Positional is the *primary* resolver;
caption is the *cross-check*, not the other way round:

1. **Positional (primary):** nth image in `_content_list.json` reading order →
   `Fig {chapter}.{n}`, where `chapter` is the leading number in the case directory name.
2. **Caption (verifier):** regex over `image_caption`, **accepted only if its chapter matches
   the directory chapter**. Where it fires, it must agree with positional.
3. Report the disagreement rate as a QA metric. On this corpus it is currently **0/96**.

**One honesty caveat, and you should say it out loud in the README:** the ground truth was
itself derived from the hand-fixed markdown, and if that fix was done positionally then the
100% is partly self-confirming. The 96 pairs independently confirmed by the caption regex are
not subject to that objection — quote *that* number when someone pushes back.

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
the `time.sleep(5)` / `time.sleep(30)` calls are working around, and it is the same force
that pushed the project onto `gemma-3-27b-it` when Google tightened free-tier limits
mid-project (§2.6). Your $20–50 estimate was ~10× too pessimistic.

**The important consequence:** that constraint scales with corpus size, and **calibration
does not run on the corpus.** 93 cases × repeated passes on a free tier is a quota problem;
30 cases once is a rounding error. Roughly:

| | Cases | Passes | Cost on a paid frontier model |
|---|---|---|---|
| Full curation run | 93 | many | $2–8 — worth optimising |
| **Calibration subset** | **30** | **1–4** | **~$0.25 per pass** |

So the decision that was genuinely forced at corpus scale is **not forced at calibration
scale**, and that is where judge quality actually matters. Carrying the free-tier constraint
into Phase 2 would be inheriting a limitation you are no longer under.

### Decision (revised — the project has moved to OpenAI)

`.env.example` and `pyproject.toml` now target the OpenAI SDK. Everything in §2.6 about Gemma
stays true as **history** — it is what produced the 51 cases and the 0-flag editor — but it is
no longer the forward plan. Three consequences:

**1. The binding constraint flips.** Free-tier quota was the dominant cost of the old setup
(hence every `time.sleep(30)`). On a paid API that constraint disappears and money becomes
real — but the token counts above still hold, so a full run is single-digit dollars. You have
traded a *wall-clock* problem for a *small budget* problem. That is a good trade, and it
retires the constraint that forced the Gemma decision in the first place.

**2. The judge must not also be OpenAI.** The self-enhancement-bias argument is about model
*family*, not vendor identity. If the extractor is an OpenAI model, `JUDGE_MODEL` has to come
from somewhere else — Anthropic, Google, or a local open-weights model. Setting both to
OpenAI recreates exactly the flaw §2.6 is trying to measure, just with a better model.

**3. Hold the extractor fixed once you pick it.** Whatever you choose, freeze it before Phase
2 starts. The 2×2 in Phase 2 step 5 measures *rubric vs judge capability*; if the extractor
drifts underneath it, none of those four cells are comparable and the experiment is wasted.

The 2×2 itself is unchanged in shape — substitute your chosen extractor for `gemma-3-27b-it`
in the left-hand column, and keep the historical Gemma numbers as the "before" baseline they
already are.

---

## 5. Repo layout

```
medical-vlm-curator/
  pyproject.toml           # pin mineru==2.5.4 (+ pipeline backend) when Phase 0.5 lands
  tests/fixtures/
    figure_ground_truth.json   # Fig->file answer key; labels + hashes only, no source
                               # text - safe to commit. 118 pairs if scraped from
                               # notebook 3's output, ~158 if rebuilt from the zip
                               # (Phase 0 step 7). NEVER generate it with figures.py.
  .gitignore               # data/, outputs/, *.pdf, *.jpg  <- non-negotiable
  README.md
  DATASHEET.md             # dataset card, Gebru et al. format
  src/curator/
    config.py              # models, thresholds, retry cap, corpus path via env var
    figures.py             # NEW - deterministic figure resolver (§3), replaces the
                           #       missing corpus-normalisation script (§1.1)
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
2. Port the notebook into `src/curator/` with **no logic changes** — but note the fidelity
   checkpoint just got weaker, and know why. The original plan was a byte-faithful port
   verified by identical output. **The SDK swap to OpenAI makes that impossible**: the model
   changes at the same time as the code, so any output difference has two possible causes and
   you cannot attribute it. Two options, in order of preference:
   - **If you still have working Google credentials:** port verbatim on the old SDK first,
     confirm one case reproduces, commit that, *then* swap the SDK as an isolated diff. This
     preserves a real checkpoint and is worth the extra hour.
   - **If you don't:** accept a structural checkpoint instead — same nodes, same edges, same
     prompt text, same parsed fields — and state in the commit message that output equivalence
     was not verified. An honest weaker check beats a fake strong one.

   **Also: the multimodal call shape differs.** The prototype uploads images with
   `client.files.upload(...)` and passes file handles (Gemini). OpenAI takes base64 image
   parts inline in the message, and there is no upload/delete lifecycle — so the
   `[client.files.delete(name=f.name) for f in client.files.list()]` cleanup lines in both
   notebooks have no OpenAI equivalent and should simply disappear. Budget an hour for the
   image path; it is the least mechanical part of the port.
3. Drop the unused LangChain import. Add `pytest` + `ruff` + one smoke test.
4. **Record VCR cassettes for 3 cases** (§5) before touching any logic. Everything in
   Phase 1 except the live-API paths then becomes testable offline and instantly —
   which is what makes a 10 hrs/week budget realistic against a rate-limited free tier.
5. **Archive the notebooks and strip their outputs.** All three carry generated clinical text
   in saved cell outputs — notebook 3 alone held ~199 KB. `.gitignore` cannot help you here;
   only `nbstripout` can. Install it as a git filter (`nbstripout --install
   --attributes .gitattributes`) *before* the first `git add` of any notebook. Keep an
   unstripped copy **outside the repo** — see step 6 for why you need it.
6. **Download `extracted_case_report_image_filtered.zip` out of Google Drive today.** This is
   the one genuinely irreplaceable artifact in the project. Per §1.1 the script that produced
   it no longer exists, so if that Drive folder is lost you cannot rebuild the corpus your
   pipeline actually consumed — from `d:/extracted_data` or from anything else. Grab
   `benchmark_image.zip` and `outputs/` while you are there; neither is on local disk.
7. **Build the figure ground truth** at `tests/fixtures/figure_ground_truth.json` (§3) and
   commit it — labels and content hashes only, no source text, so it is safe to publish.
   Three valid routes and one invalid one:

   | Route | Coverage | Notes |
   |---|---|---|
   | Re-run cell 52's regex over **all 93** markdowns in the zip | ~158 pairs | **Best.** Deterministic, no LLM. |
   | Scrape notebook 3's saved cell-52 output | 118 pairs | Already done — in the session scratchpad. Truncated: cell 52 only iterated the 78 rows surviving `dropna()`. |
   | Hash filenames in `benchmark_image.zip` (`Fig. 75.1.jpg` → source hash) | 70 pairs | Independent check on the copy step. |
   | ~~Generate it with `figures.py`~~ | — | **Invalid — circular.** You would be scoring the resolver against its own output. Do not do this, however convenient it looks later. |

   **If the Drive copy is gone entirely,** `figures.py` is still testable: positional and
   caption resolution are mutually independent, so they cross-validate (96 confirmed pairs,
   0 disagreements — §3). You lose the external key, not the test. For the 24 figures with no
   parseable caption, hand-label from `_origin.pdf` / `_layout.pdf`, which are on local disk.

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
5. **Separate the rubric effect from the model effect.** This is the highest-value hour in
   the whole plan and it costs about a dollar. You have two suspects for the 0% flag rate
   (§2.6) and 30 labelled cases; that is enough for a 2×2:

   | | `gemma-3-27b-it` | capable judge (different family) |
   |---|---|---|
   | **Empty rubric** | the historical baseline — reproduce it | isolates model capability |
   | **Fixed rubric** | isolates the prompt fix | the intended configuration |

   Four runs × 30 cases ≈ 4 × $0.25. Report κ for each cell. Whatever the answer, you can
   say something precise: *"the prompt fix bought X, the model bought Y."* If the model
   dominates, that is a finding about small-model self-critique worth writing up. If the
   rubric dominates, you have demonstrated that a free model is sufficient given a correct
   prompt — which is a better result for anyone who has to run this on a budget.

   If you have to cut something, cut the empty-rubric/capable-judge cell. Keep the diagonal.

   **The quota constraint that forced Gemma does not apply here.** It bit because you were
   running 93 cases repeatedly on a free tier. Calibration is 30 cases, once. At that scale a
   paid frontier judge is well under $1 (§4) — so the decision you were pushed into
   corpus-wide is simply not forced at calibration scale. Don't inherit the constraint out of
   habit.
6. **Report the funnel honestly.** You no longer have to guess at it — notebook 2's saved
   outputs give you the real numbers for the first pipeline run:

   | Stage | N | Note |
   |---|---|---|
   | Case directories in corpus | 93 | |
   | Produced a parseable grading | 85 | 8 returned `"I can't."` or `None` |
   | Passed soft filter (all scores ≥ 3) | 72 | |
   | Passed hard filter (all scores ≥ 4) | 52 | this is the gate that ran |
   | Passed radical filter (all scores ≥ 5) | **0** | nothing in the corpus is a 5 |
   | Reached extraction | 51 | 1 lost between filter and loop — **unexplained** |
   | Passed the editor | 51 / 51 | see §2.6 — the editor flags nothing |

   Mean grader scores across the 85: case presentation **4.00**, integrative reasoning
   **3.94**, transparency **3.80**, image usefulness **4.42**; 82/85 had an explicit
   differential, 85/85 a stated final diagnosis.

   Two things to notice, both worth a paragraph in the README. **First, the radical filter
   returns zero** — the grader never awards 5 on all four axes, so its effective range is
   3–4 and that compresses any κ you compute. Say so before someone asks. **Second, image
   usefulness scored highest of all four criteria (4.42) — and §2.1 shows 13 of 93 cases had
   no images resolved at all.** The grader was scoring the usefulness of images it never
   received. Check that specific correlation during calibration; it is the cleanest evidence
   of ungrounded scoring you are likely to get, and it is exactly the kind of finding that
   makes a calibration report worth reading.

   Then produce the same table for your own run: N PDFs → passed grading → extracted →
   passed editor → final, with **every case in exactly one terminal state**. "93 cases" in a
   filename is not a yield.

### Phase 0.5 — Reproducible extraction (deferred to after Phase 2)
Moved later deliberately: it adds **zero capability** right now (the corpus already exists),
pulls torch + CUDA (~2–3 GB) into a working environment, and its value is reproducibility
*documentation* — a Phase 4 concern. Do not let it block calibration.

`notebooks/archive/01_pdf_extraction_mineru.ipynb` is the whole extraction stage, and it is
six lines: `pip install -U "mineru[core]"`, `mineru -p raw/ -o processed/`, prune, zip,
download. Reviewing it against the corpus turns up three problems you need to know about
**before** you rely on it:

1. **It does not reproduce your corpus.** Its saved output shows it installed
   **mineru 2.7.6** and logged *"Using transformers as the inference engine for VLM"* — the
   VLM backend. Your golden corpus is **2.5.4, `_backend: pipeline`** in all 93
   `_middle.json`. Different version *and* different backend. This notebook is a later,
   unrelated run — not the recipe that produced `d:/extracted_data`.
2. **`pip install -U` pins nothing.** Re-running it today gets whatever is newest. Any
   `extract.py` must pin `mineru==2.5.4` and pass the pipeline backend explicitly.
3. **Its cell 4 deletes exactly the file `figures.py` depends on.** It removes everything
   except `.md` and `images/` — including `_content_list.json`, which is the sole input to
   the figure resolver in §3. If you re-run this notebook as written, you destroy the
   resolver's data source. **Delete that cell before ever running it again.**

#### The refactor: `.py` in git, data in the cloud, notebook as a five-line driver

MinerU needs a GPU, so this stage stays in Colab or Kaggle — but nothing except the *driving*
has to live in the notebook. The split is **logic in git, data in Drive/Datasets, GPU in the
cloud**, and it is what makes the stage reviewable and testable at all.

`src/curator/extract.py` **shells out to the MinerU CLI** rather than importing its Python
API. The CLI is the stable interface across versions, and it is what produced the golden
corpus. It also means the wrapper is testable locally with no GPU — mock the subprocess and
assert on the argv you built.

```toml
# pyproject.toml — optional, so a local `pip install -e .` stays torch-free
[project.optional-dependencies]
extract = ["mineru[core]==2.5.4"]
```

The notebook then reduces to:

```python
!pip install -q "medical-vlm-curator[extract] @ git+https://github.com/<you>/medical-vlm-curator.git"
from google.colab import drive; drive.mount('/content/drive')
!python -m curator.extract     --pdf-dir  /content/drive/MyDrive/Project_Medical_LMM/raw_case_report     --out-dir  /content/processed     --backend  pipeline
!zip -qr /content/processed.zip /content/processed
!cp /content/processed.zip /content/drive/MyDrive/Project_Medical_LMM/
```

Commit that notebook (stripped — the `nbstripout` filter from Phase 0 step 5 handles it). It
is documentation of the stage, and it contains no data.

**Prerequisite: the repo must be on GitHub.** `pip install git+https://…` needs a remote, and
there isn't one yet. Public is fine and is already the plan — `.gitignore` keeps every PDF and
every generated case out, so pushing code exposes nothing. A PAT-authenticated `git clone`
works if you want to stay private first, but public is simpler.

#### Colab vs Kaggle

| | Colab | Kaggle |
|---|---|---|
| Input data | `drive.mount` | private **Dataset**, read-only at `/kaggle/input` |
| Output | ephemeral — copy to Drive | `/kaggle/working`, savable as a versioned Dataset |
| Internet | on | **off by default** — toggle it on or `pip` fails with a confusing DNS error |
| Python | 3.12 | 3.11 — both satisfy `requires-python = ">=3.11"` |
| Session | ~12 h, T4 | ~12 h, P100 or 2×T4 |

**Prefer Kaggle**, marginally. A private Dataset is versioned and the notebook declares its
input explicitly, which is better provenance than a Drive path that gets reorganised — and
"copyrighted PDFs in a private, versioned Dataset attached to the run that used them" is a
materially better governance story for the DATASHEET than "in my personal Drive".

#### Three things to get right in `extract.py`

1. **Pin `2.5.4` and pass the backend explicitly.** Verify the exact flag name for that
   release — the backend options changed across the 2.5→2.7 series, which is precisely how
   the archived notebook ended up on VLM without anyone choosing it.
2. **Never delete `_content_list.json`.** Old cell 4 pruned everything except `.md` and
   `images/`. If you want a slim zip, make pruning a separate opt-in `--prune` flag, and have
   it refuse to run unless `figure_ground_truth.json` has already been built.
3. **Cache the model weights.** MinerU pulls ~2 GB on every cold session (the saved output
   shows `Fetching 14 files` and an 810 MB MFR model). Point its cache at Drive or a Kaggle
   Dataset and you save several minutes per run.

#### Then verify

Re-run on 5 `_origin.pdf` files and diff against existing output. **If it diverges, do not
regenerate the corpus** — treat `d:/extracted_data` as the golden reference and document the
divergence. Regenerating invalidates any calibration done against it.

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

- **Phase 1:** `tests/test_figures.py` runs against
  `tests/fixtures/figure_ground_truth.json` and asserts: **every pair in the fixture resolves
  correctly** (118/118 on the scraped key; re-baseline if you rebuild the larger one),
  **0 disagreements where positional and caption both fire** (currently 0/96), and
  **0 unlabelled images sent to the model**. Assert against `len(fixture)`, not a hardcoded
  118, so rebuilding the key doesn't silently weaken the test. Real labels, not a snapshot —
  treat any drop as a build failure.
- **Phase 1:** run the graph on 3 cases and confirm the resolved figure map matches ground
  truth **when built from raw `d:/extracted_data`**, not from the pre-fixed corpus. That is
  the actual regression being closed (§1.1) — the old path only worked on the hand-edited zip.
- **Phase 0.5 (deferred):** MinerU re-run on 5 PDFs diffs clean against the golden copy,
  pinned to `2.5.4` with the pipeline backend.
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
- **The editor may calibrate to κ ≈ 0** (§2.6 — it has never rejected anything). Budget for
  this being the *result* rather than a blocker: report it, run the 2×2 in Phase 2 step 5 to
  attribute it between the empty rubric and the quota-forced model, then re-measure. A
  before/after on a judge you diagnosed and repaired beats a single good number.
- **Don't over-fit the plan to `gemma-3-27b-it`.** It is in the code because free-tier limits
  changed mid-project, not because it was chosen. Treat the model as a swappable parameter in
  `config.py` from day one, and make sure nothing in the prompts or parsing silently depends
  on its quirks — otherwise you bake a quota decision from last year into the architecture.
- **The grader's effective range is 3–4, not 1–5** (Phase 2 funnel: zero cases score 5 on all
  axes). Range restriction depresses κ mechanically. Know this before you compute it, and
  report the score distribution alongside the agreement statistic.
- **The corpus-normalisation script is missing** (§1.1). Until `figures.py` replaces it, the
  pipeline's input is a zip file you cannot rebuild. Treat that zip as irreplaceable until
  Phase 1 lands — back it up off Drive.
