# MURL final professor talking points

- **Updated:** November 15, 2026
- **Status:** v0.8 study complete; no held-out rerun or post-test tuning
- **Selected system:** Qwen3-4B v0.6 hierarchical multi-task

## Executive summary

MURL tested whether supervised parameter-efficient fine-tuning improves structured grading of text transcripts of worked Calc I–II integration solutions. The final system returns a correctness label, first-error step, error taxonomy, evidence, feedback, and confidence in validated JSON.

Selection used only the frozen 48-example validation split. Qwen3-4B v0.6 was selected under precommitted validity, correctness, taxonomy, and localization gates. It was then evaluated exactly once on 47 held-out examples with one generation pass, no retries or repair, and one scoring invocation.

## Final held-out result

| Measure | Result |
| --- | ---: |
| Schema-valid | 47/47 |
| Correctness accuracy | 39/47 (0.8298) |
| Correctness macro-F1 | 0.8445 |
| Taxonomy macro-F1 | 0.7037 |
| First-error exact | 24/26 (0.9231) |

The strongest operational result is reliable structured output. The main remaining diagnostic weakness is substitution: all four held-out gold substitution errors were assigned another label.

## Study progression

| Phase | Main finding |
| --- | --- |
| Prompt baselines | Few-shot improved Qwen2.5 correctness over zero-shot, but format and localization failures remained. |
| Initial QLoRA | Fine-tuning largely solved JSON validity; label selection became the main bottleneck. |
| Guidance and sampling | Added definitions and taxonomy balancing produced tradeoffs between correctness and diagnosis. |
| Hierarchical objectives | Qwen3 v0.6 gave the best eligible balance of correctness, taxonomy, and localization. |
| Cross-family audit | Qwen3, Phi, and historical Qwen2.5 arms were recalculated from archived outputs under one rule. |
| One-time test | The frozen Qwen3 v0.6 winner generalized descriptively to the 47-example held-out split. |

## Dataset and protocol

| Item | Value |
| --- | --- |
| Benchmark | 250 authored student-style integration solutions |
| Problem families | 34, kept disjoint across splits |
| Topics | 8 Calc I–II integration topic groups |
| Splits | 155 train / 48 validation / 47 test |
| Output | Correctness, first error, taxonomy, evidence, feedback, confidence |
| Compute | QLoRA on Google Colab Tesla T4 |
| Decision role | Instructor assistance only; not autonomous grading |

The benchmark contains authored text, not real student submissions or OCR. Taxonomy supports are sparse, and several per-label results depend on one or two examples. Claims therefore apply to this benchmark rather than classroom populations.

## Closeout review

The 60 records originally queued for second review were independently checked against the annotation guide. All existing label triples were confirmed and recorded in a separate provenance ledger; the frozen dataset was not edited. The closeout was AI-assisted, so faculty sign-off is still required before calling it blinded human adjudication.

All 47 held-out feedback strings were also reviewed with a four-part 0–2 rubric for mathematical correctness, specificity, actionability, and tone:

| Feedback result | Count |
| --- | ---: |
| 8/8 | 25 |
| 7/8 | 18 |
| Below 7/8 | 4 |

The four material feedback failures missed a required back-substitution, rejected a valid alternative method, incorrectly advised dropping a constant integrand, or incorrectly required changed bounds after back-substitution. These cases reinforce the human-oversight requirement.

## What is complete

- Versioned benchmark, annotation guide, schema, and evaluator.
- Prompt-only baselines and matched base controls.
- QLoRA smoke test and multiple full training recipes.
- Cross-family validation audit and frozen winner selection.
- Exactly one held-out generation and scoring pass.
- Final reports, figures, notebooks, hashes, and archive-only reproduction.
- Post-study second-review ledger and feedback-quality rubric.
- Durable Drive backup of the ignored run archives.

## What must not happen

- Do not regenerate or rescore held-out outputs as a new evaluation.
- Do not tune prompts, labels, thresholds, or adapters using the held-out result.
- Do not claim OCR, handwriting, real-student, or autonomous-grading validity.
- Do not present the AI-assisted closeout rubric as blinded human evaluation.

Any future modeling work starts a separately versioned study with a new untouched test set.

## Professor discussion

1. Is faculty sign-off desired for the 60-record resolution ledger and the 47-output feedback rubric?
2. Should the next study prioritize real student transcripts, substitution-error coverage, or OCR robustness?
3. What evidence standard is required before classroom pilot use?
4. Should the benchmark receive an explicit public license, or remain repository-local?

## Authoritative references

- [Final scientific report](../reports/MURL-v0.8-final-scientific-report.pdf)
- [Final handoff](../reports/v0.8-final-handoff.md)
- [One-time held-out report](../reports/v0.8-final-test-report.md)
- [Cross-family selection audit](../reports/v0.8-final-cross-family-report.md)
- [Second-review closeout](../reports/v0.8-second-review-closeout.md)
- [Feedback-quality review](../reports/v0.8-feedback-quality-review.md)
- [Dataset card](data-card-v0.1.md)
