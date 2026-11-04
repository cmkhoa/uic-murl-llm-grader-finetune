# Experiment report index

This index covers the v0.1–v0.8 experiment, validation, review, planning, and
final reports. Individual completed experiment reports use the same reading
order where their material supports it: status, question, frozen setup,
results, interpretation, limitations, artifacts, and decision. Plans retain
their prospective structure, and final reports retain their audit structure.

Metric denominators are shown when they are central to interpretation.
Validation task metrics are generally valid-output-only unless a report says
otherwise; eligibility checks use the stated all-example or eligible-example
denominator. Historical percentage formatting is preserved to avoid changing
reported results. The [v0.8 final cross-family audit](v0.8-final-cross-family-report.md)
and [one-time held-out report](v0.8-final-test-report.md) are authoritative for
overlapping final values.

| Report | Status | Model or recipe | Primary result |
| --- | --- | --- | --- |
| [v0.1 validation prompt baselines](v0.1-validation-baselines.md) | Completed; few-shot advanced | Qwen2.5-3B, zero-shot vs few-shot | Both were 45/48 valid; few-shot correctness was 51.1% vs 28.9%. |
| [v0.1 prompt selection](v0.1-prompt-selection.md) | Completed; prompt frozen | Qwen2.5-3B few-shot | Few-shot won the predeclared ordering and was authorized for one held-out run. |
| [v0.1 held-out baseline](v0.1-heldout-test-baseline.md) | Completed; baseline only | Qwen2.5-3B few-shot | 40/47 valid; 60.0% correctness on valid outputs. |
| [v0.1 QLoRA smoke](v0.1-qlora-smoke.md) | Completed; infrastructure passed | Qwen2.5-3B, 10-step QLoRA | 32/48 valid; 0/15 first-error exact; not eligible for held-out use. |
| [v0.1 full QLoRA validation](v0.1-qlora-full-validation.md) | Completed; review required | Qwen2.5-3B, 60-step QLoRA | 48/48 valid; 58.3% correctness and 31.7% taxonomy macro F1. |
| [v0.1 full QLoRA error review](v0.1-qlora-full-error-review.md) | Completed; v0.2 change selected | Qwen2.5-3B v0.1 adapter | Formatting was solved; label selection and localization remained the main errors. |
| [v0.2 label-guidance validation](v0.2-label-guidance-validation.md) | Completed; mixed result | Qwen2.5-3B, label-guidance QLoRA | 47/48 valid; correctness rose to 66.0%, while taxonomy macro F1 fell to 20.7%. |
| [v0.3 taxonomy-guidance validation](v0.3-taxonomy-guidance-validation.md) | Completed; not selected | Qwen2.5-3B, taxonomy-guidance QLoRA | 47/48 valid; correctness reached 72.3%, but taxonomy macro F1 was 19.7%. |
| [v0.4 taxonomy-balance validation](v0.4-taxonomy-balanced-validation.md) | Completed; not promoted | Qwen2.5-3B, equal-taxonomy oversampling | 46/48 valid; taxonomy macro F1 was 32.7% and first-error exact was 20/28, but correctness fell to 27/48 overall. |
| [v0.4 cross-family plan](v0.4-cross-family-comparison-plan.md) | Draft; not executed as written | Proposed frozen cross-family protocol | Required a satisfactory frozen recipe before execution. |
| [v0.5 moderated-balance validation](v0.5-moderated-taxonomy-balance-validation.md) | Completed; Qwen2.5 nominee | Qwen2.5-3B, moderated taxonomy balance | 47/48 valid; 30/48 correct overall; taxonomy macro F1 0.3201; utility 0.4936. |
| [v0.5 validation error review](v0.5-validation-error-review.md) | Completed | Qwen2.5 v0.3–v0.5 comparison | v0.5 was the best balanced recipe; taxonomy remained the main bottleneck. |
| [v0.6 hierarchical multi-task plan](v0.6-hierarchical-multitask-plan.md) | Executed; see validation report | Qwen2.5-3B, staged 4× field weighting | Precommitted at least 47/48 validity plus a utility gain over v0.5. |
| [v0.6 hierarchical multi-task validation](v0.6-hierarchical-multitask-validation.md) | Completed; failed eligibility | Qwen2.5-3B, staged 4× field weighting | Utility rose to 0.5137, but validity was 46/48 and taxonomy macro F1 was 0.2449. |
| [v0.7 two-stage diagnosis plan](v0.7-two-stage-diagnosis-plan.md) | Executed; see validation report | Qwen2.5-3B, ordinary SFT then 2× diagnosis | Required all five v0.5 replacement gates. |
| [v0.7 two-stage diagnosis validation](v0.7-two-stage-diagnosis-validation.md) | Completed; failed taxonomy gate | Qwen2.5-3B, ordinary SFT then 2× diagnosis | 48/48 valid and utility 0.5002, but taxonomy macro F1 0.3029 missed the 0.3201 floor. |
| [v0.8 cross-family execution record](v0.8-cross-family-execution-plan.md) | Executed; superseded by final audit | Qwen3-4B and Phi-4-mini, v0.5–v0.7 | Qwen3 v0.6 was the highest-utility eligible arm; several other families were infeasible or unscored. |
| [v0.8 final cross-family report](v0.8-final-cross-family-report.md) | Final validation audit | Qwen2.5, Qwen3, and Phi recipe comparison | Qwen3-4B v0.6 was selected at 47/48 valid, 35/48 correct, taxonomy macro F1 0.6984, and utility 0.7020. |
| [v0.8 final test report](v0.8-final-test-report.md) | Final one-time held-out evaluation | Qwen3-4B v0.6 | 47/47 valid; 39/47 correct; taxonomy macro F1 0.7037; first-error exact 24/26. |
| [v0.8 final handoff](v0.8-final-handoff.md) | Complete; no rerun or tuning | Qwen3-4B v0.6 | Records the final selection, held-out result, reproduction links, and prohibited next actions. |

## Cross-report consistency note

The final v0.8 audit recalculated utilities from unrounded archived aggregates.
For Qwen3 v0.5, rounded display components produce 0.6226, while the unrounded
components produce the authoritative 0.6225. The execution record now states
both values and uses 0.6225 in its comparison table. This precision difference
does not affect eligibility, ranking, or the selected winner.
