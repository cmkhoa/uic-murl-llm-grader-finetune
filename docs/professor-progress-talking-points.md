# MURL progress meeting talking points

- **Date:** October 29, 2026
- **Focus:** LLM-assisted grading of integration solutions

## Main takeaways

- Reproducible benchmark + five completed QLoRA variants
- Strongest gain: structured-output reliability
- Main tradeoff: correctness vs. error classification and localization
- Current compromise: v0.5; final model selection pending

## Completed progress

| Area | Status |
| --- | --- |
| Task | Correctness label, first error step, error category, evidence, feedback, confidence |
| Inputs | Integration problem + reference solution + student-style text solution |
| Benchmark | 250 authored examples; 34 problem families; eight topics |
| Splits | 155 train / 48 validation / 47 test; no shared problem families |
| Infrastructure | Versioned prompts; schema checks; automated scoring; saved adapters and run records |
| Baselines | Zero-shot and few-shot; selected few-shot baseline tested once |
| Fine-tuning | Smoke test + five full variants; evaluation on 48 validation examples; no fine-tuned model tested on the 47 test examples yet |

## Example data point

- **Source:** [Training dataset](../data/processed/train.jsonl); `int_sub_0003`
- **Metadata:** authored example; substitution; easy; single-annotator review
- **Problem:** ∫ 3x(x² + 4)⁴ dx

| Step | Student work | Reference solution |
| --- | --- | --- |
| 1 | u = x² + 4; **du = x dx** | u = x² + 4; **du = 2x dx** |
| 2 | 3 ∫ u⁴ du | (3/2) ∫ u⁴ du |
| 3 | 3u⁵/5 = 3(x² + 4)⁵/5 + C | (3/10)u⁵ + C |
| 4 | — | (3/10)(x² + 4)⁵ + C |

| Reference grading field | Value / explanation |
| --- | --- |
| Overall correctness | `incorrect`; missing factor of 2 affecting the final answer |
| First error step | `1`; incorrect differential conversion |
| Error taxonomy | `substitution` |
| Feedback talking point | du = 2x dx; corrected answer: 3(x² + 4)⁵/10 + C |

- **Metric illustration:** predicted `incorrect` → correct overall label; predicted `substitution` → correct taxonomy; predicted step `1` → exact localization match
- **Later-step prediction:** step `2` or `3` → no first-error exact match; downstream consequence of step `1`
- **Labels above:** dataset annotations; not model predictions

## Techniques used

| Technique | Implementation / purpose |
| --- | --- |
| Supervised fine-tuning | Qwen2.5-3B-Instruct; canonical JSON grading targets |
| QLoRA | 4-bit NF4 base model + trainable low-rank adapters; Colab Tesla T4 |
| Training settings | Rank 8; alpha 16; dropout 0.05; 60 steps; effective batch 8; learning rate 0.0002 |
| Label guidance | v0.2: explicit correctness and taxonomy definitions |
| Taxonomy guidance | v0.3: decision rules for commonly confused error categories |
| Equal resampling | v0.4: 68 records per category; 680 total |
| Moderated resampling | v0.5: 68 no-error records + 25 per other category; 293 total |
| Experimental controls | Fixed model revision, seed, validation split, greedy decoding; matched base controls |

- Resampling: training split only; repeated examples, no new independent data
- Macro F1: equal weight per class

## What changed at each step

- **Separate training runs:** fresh adapter from the same base model for each version; no successive training of one adapter through all five versions
- **Fixed across full runs:** base-model revision, seed, QLoRA settings, 60 optimizer steps, decoding, and 48-example validation split
- **Prompt changes:** instructions used during both training and evaluation; new adapter training for each prompt variant

### Prompt baselines — examples in the prompt

- **Zero-shot:** grading instructions, allowed labels, output schema; no demonstrations
- **Few-shot addition:** three training examples with reference outputs — correct work, incorrect differential conversion, insufficient work
- **Learning mechanism:** examples in the inference context; no parameter updates
- **Validation result:** same validity, 45/48; correctness accuracy 28.9% → 51.1%; correctness macro F1 18.7% → 29.1%
- **Selection:** few-shot; one subsequent held-out test evaluation
- **Remaining problem:** overprediction of `correct`; JSON-escaping failures; weak first-error localization

### v0.1 — learning the grading output through fine-tuning

- **Change:** first full QLoRA adapter training on 155 training conversations
- **Training input:** short grading instruction + problem + reference solution + student steps
- **Training target:** canonical JSON grading response; training loss on assistant-response tokens only
- **Parameter updates:** low-rank adapters; frozen quantized base-model weights
- **Budget:** 60 optimizer steps; batch 1 × gradient accumulation 8; approximately three training-data passes
- **Matched comparison:** untouched base vs. trained adapter under the same SFT prompt
- **Result:** 0/48 → 48/48 schema-valid outputs; adapter correctness accuracy 58.3%
- **Failure review:** 10 of 19 gold `incorrect` cases labeled `minor_error`; neither of two `insufficient_work` cases identified; 10 `substitution` predictions vs. one gold case
- **Next hypothesis:** missing label definitions contributing to grading-label confusion
- **Comparison limit:** different prompt from few-shot; no isolated fine-tuning-effect claim against the earlier few-shot test result

### v0.2 — explicit definitions and output consistency rules

- **Changed component:** system prompt; regenerated training conversations with expanded instructions
- **Unchanged:** 155 source examples, target annotations, model and training settings
- **Added guidance:**

| Addition | Specific instruction / distinction |
| --- | --- |
| `correct` | Valid material work and final answer |
| `minor_error` | Local, non-conceptual mistake; method substantially intact after correction |
| `incorrect` | Material error affecting method, result, or validity |
| `insufficient_work` | Too little work for defensible step-level judgment; null error step |
| Ten taxonomy definitions | Explicit meanings for algebra, sign, antiderivative, substitution, and remaining categories |
| Consistency rules | `correct` → null error step + `none`; `minor_error` / `incorrect` → earliest invalid step + non-`none` taxonomy |

- **Result vs. v0.1:** correctness accuracy 58.3% → 66.0%; correctness macro F1 43.7% → 49.3%
- **Cost:** taxonomy macro F1 31.7% → 20.7%; validity 48/48 → 47/48
- **Remaining confusions:** algebra vs. antiderivative; sign vs. substitution; bounds vs. other
- **Interpretation:** better correctness-label selection; unresolved error-category discrimination

### v0.3 — decision rules for confusing error categories

- **Changed component:** extra paragraph in the v0.2 prompt; all v0.2 definitions retained
- **Unchanged:** same 155 examples and targets; same training budget
- **Core rule:** category based on the earliest error’s mechanism, not the problem’s topic

| Error mechanism | Added distinction |
| --- | --- |
| Invalid variable change, differential conversion, back-substitution | `substitution`; not every error in a substitution problem |
| Wrong integration operation | `antiderivative` |
| Wrong simplification, expansion, factoring, arithmetic | `algebra`; distinct from integration-rule errors |
| Introduced or lost sign | `sign` |
| Limit or definite/improper-integral evaluation error | `bounds` |

- **Result vs. v0.2:** correctness accuracy 66.0% → 72.3%; correctness macro F1 49.3% → 53.0%
- **Unmet objective:** taxonomy macro F1 20.7% → 19.7%; first-error exact 53.6% → 50.0%
- **Next hypothesis:** category imbalance worth testing after two guidance revisions without taxonomy improvement

### v0.4 — equal representation of error categories

- **Prompt choice:** return to v0.2; removal of v0.3’s extra decision-rule paragraph
- **Controlled reference:** v0.2; sampling as the changed factor relative to that version
- **Sampling rule:** every taxonomy category expanded to 68 records, the largest original category count
- **Implementation:** seeded shuffled cycles within each category; every source example included before repetition; final shuffle across categories
- **Unchanged targets:** duplicated existing conversations; no invented examples or relabeling

| Taxonomy | Original count / v0.2 | v0.4 count | v0.5 count |
| --- | ---: | ---: | ---: |
| none | 68 | 68 | 68 |
| antiderivative | 25 | 68 | 25 |
| sign | 15 | 68 | 25 |
| substitution | 13 | 68 | 25 |
| algebra | 13 | 68 | 25 |
| other | 7 | 68 | 25 |
| bounds | 6 | 68 | 25 |
| integration_by_parts | 3 | 68 | 25 |
| trig_identity | 3 | 68 | 25 |
| partial_fractions | 2 | 68 | 25 |
| **Total** | **155** | **680** | **293** |

- **Exposure shift:** `none` share 43.9% → 10.0%; each partial-fractions source example repeated 34 times
- **Result vs. v0.2:** taxonomy macro F1 20.7% → 32.7%; first-error exact 53.6% → 71.4%
- **Cost vs. v0.2:** correctness accuracy 66.0% → 58.7%; validity 47/48 → 46/48
- **Concern:** less no-error exposure, changed correctness-label distribution, heavy repetition of rare examples
- **Interpretation:** useful diagnostic gains; weaker overall correctness; no established causal explanation from one run

### v0.5 — less aggressive category balancing

- **Changed component:** sampling threshold only relative to v0.4; same v0.2 prompt
- **New rule:** minimum 25 records per category; no downsampling of categories already above 25
- **Threshold rationale:** 25 = largest original non-`none` category count; derived from training data
- **Resulting mix:** 68 `none` + nine categories × 25 = 293 records; all 155 unique examples retained
- **Exposure shift vs. v0.4:** `none` share 10.0% → 23.2%; maximum source repetition 34 → 13
- **Intended benefit:** more no-error representation and less rare-example repetition, while retaining category balancing
- **Result vs. v0.4:** correctness accuracy 58.7% → 63.8%; correctness macro F1 45.5% → 47.9%; validity 46/48 → 47/48
- **Tradeoff vs. v0.4:** taxonomy macro F1 32.7% → 32.0%; first-error exact 71.4% → 53.6%
- **Current interpretation:** compromise candidate; below v0.3 correctness and v0.4 localization; no across-the-board winner

### Training exposure caveat

| Recipe | Records in training file | Fixed effective record budget | Approximate passes |
| --- | ---: | ---: | ---: |
| Original sampling / v0.2 | 155 | 480 | 3.10 |
| Equal sampling / v0.4 | 680 | 480 | 0.71 |
| Moderated sampling / v0.5 | 293 | 480 | 1.64 |

- **Equal optimizer budget:** 60 steps × effective batch 8
- **Unequal exposure:** different repetition frequencies and dataset passes; not equal epochs or token counts
- **v0.4 caveat:** all sources retained in the file; no guarantee of every source consumed within 480 sampled records

- **Key distinction:** v0.4 and v0.5 based on v0.2 guidance, not cumulative additions to v0.3
- **Sampling changes only:** unchanged source examples and target labels; changed repetition frequencies

## Metric definitions

| Metric | Meaning | Example / interpretation |
| --- | --- | --- |
| Correctness accuracy | Share of valid outputs with the correct overall grading label: `correct`, `minor_error`, `incorrect`, or `insufficient_work` | 30 matching labels / 47 valid outputs = 63.8%; agreement with reference labels, not share of correct student solutions |
| Precision | Among predictions of a label, proportion actually belonging to that label | Predicted sign errors: fraction truly sign errors |
| Recall | Among examples belonging to a label, proportion detected | Actual sign errors: fraction identified |
| F1 | Balance of precision and recall; F1 = 2 × precision × recall / (precision + recall) | High score requiring both few false positives and few missed cases |
| Macro F1 | Mean of per-label F1 scores; equal weight per label | Visibility into rare-label performance |
| Correctness macro F1 | Macro F1 across the four overall grading labels | Balanced performance across correctness classes |
| Taxonomy macro F1 | Macro F1 across error categories, including `none` | Balanced performance across algebra, sign, substitution, and other categories |
| First-error exact match | Share of eligible solutions with the first erroneous step identified exactly | Reference step 3 + prediction step 3: match; prediction step 4: no match |

## Validation results

- Same 48-example validation split
- Quality metrics: valid outputs only
- First-error exact match: evaluator’s eligible subset

| Version | Change | Valid outputs | Correctness accuracy | Correctness macro F1 | Taxonomy macro F1 | First-error exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v0.1 | Initial fine-tuning | 48/48 | 58.3% | 43.7% | 31.7% | 51.7% |
| v0.2 | Label definitions | 47/48 | 66.0% | 49.3% | 20.7% | 53.6% |
| v0.3 | Taxonomy rules | 47/48 | **72.3%** | **53.0%** | 19.7% | 50.0% |
| v0.4 | Equal sampling | 46/48 | 58.7% | 45.5% | **32.7%** | **71.4%** |
| v0.5 | Moderated sampling | 47/48 | 63.8% | 47.9% | 32.0% | 53.6% |

## Current findings

- **Format reliability:** matched base controls 0/48 valid → adapters 46–48/48 valid
- **Base-control interpretation:** format failure; mathematical quality unavailable
- **Best correctness:** v0.3
- **Best taxonomy and localization:** v0.4
- **v0.5 compromise:** partial correctness recovery; most of v0.4’s taxonomy performance retained
- **Sampling caveat:** fixed 60-step budget; different dataset exposure across recipes
- **Evidence strength:** preliminary; causation and generalization unestablished

## Limitations

- Small pilot; single seed; sparse rare-category validation examples
- Repeated validation-guided iteration; potential validation overfitting
- Authored examples only; no real student submissions
- 60 records awaiting second review
- Earlier few-shot test vs. current adapter validation: no direct comparison
- Fine-tuned test evaluation: pending
- Scope: clean text, instructor assistance; OCR and classroom reliability untested

## Next steps and professor discussion

| Priority | Talking point / decision |
| --- | --- |
| Failure review | v0.3–v0.5 confusions by taxonomy, topic, difficulty; correctness vs. localization |
| Selection criteria | Correctness priority vs. diagnostic quality vs. minimum thresholds for both |
| Threshold mismatch | Draft correctness macro F1 minimum: **49.3%**; v0.5: **47.9%**; currently below threshold |
| Cross-model comparison | Frozen recipe and selection rule first; Qwen / Llama / Phi comparison planned |
| Data quality | Second annotation review; more independent examples for rare errors |
| Final evaluation | Validation-based selection → one frozen model → one held-out test evaluation |

## Supporting records

- [Latest results and comparison](../reports/v0.5-moderated-taxonomy-balance-validation.md)
- [Cross-model protocol and selection rule](../reports/v0.4-cross-family-comparison-plan.md)
- [Dataset and review limitations](data-card-v0.1.md)
- [QLoRA setup and matched control](../reports/v0.1-qlora-full-validation.md)
- [Earlier held-out prompt baseline](../reports/v0.1-heldout-test-baseline.md)
