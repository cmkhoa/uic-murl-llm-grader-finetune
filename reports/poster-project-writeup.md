# MURL project write-up

MURL explored whether a small open-weight LLM could grade worked Calc I–II integration solutions. Given a problem, reference answer, and student steps, the model predicted correctness, the first wrong step, an error type, and student-facing feedback in structured JSON. I created a 250-example benchmark covering seven integration methods, split into 155 training, 48 validation, and 47 test examples.

Using QLoRA, I ran seven Qwen2.5-3B experiments: v0.1 established the baseline; v0.2 added label definitions; v0.3 added taxonomy rules; v0.4 balanced error categories equally; v0.5 moderated that balance; v0.6 emphasized diagnostic fields with hierarchical multi-task training; and v0.7 added a diagnosis-focused second stage. A final cross-family comparison tested these approaches on Qwen3-4B and Phi-4-mini-instruct.

The selected model was Qwen3-4B with the v0.6 recipe. On the held-out test, it produced 47/47 valid outputs, 39/47 correct grades, 0.7037 taxonomy macro-F1, and 24/26 exact first-error locations. Because four feedback responses still contained meaningful mathematical problems, instructors should retain final authority. Next steps would include real de-identified student work, faculty-reviewed labels, broader error coverage, and eventually handwritten-math support in a human-in-the-loop grading tool.
