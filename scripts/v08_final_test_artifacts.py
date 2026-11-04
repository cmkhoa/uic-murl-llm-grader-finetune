#!/usr/bin/env python3
"""Reproduce final-test tables and figures from archived outputs only.

This helper never loads a model, runs generation, or invokes the evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/murl-matplotlib-final-test")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "qwen3-4b-murl-v0.8-v0.6-final-test-20261103"
FINAL_ZIP = ROOT / "runs/final" / f"{RUN_ID}.zip"
RAW_ZIP = ROOT / "runs/final" / f"{RUN_ID}-raw-pre-score.zip"
EXECUTED_NOTEBOOK = ROOT / "runs/final" / f"{RUN_ID}-executed.ipynb"
VALIDATION_ZIP = ROOT / "runs/baseline/qwen3-4b-v08-v06-validation-results.zip"
OUT = ROOT / "reports/v0.8-final-test"
FIG = OUT / "figures"
EXPECTED = {
    "final_zip": "a14ef576c8becd41de3f38ffd9edc7a9adfc6b267679cfa21e28e97a757c2d9c",
    "raw_zip": "4d955c8a94b5031736a1647d7b16bf837e2589b6c4322a835d245b2ca2e14928",
    "executed_notebook": "8a42c363415865df1be098b6bad0c769782327899907c9f4005a456e68c60f73",
    "validation_zip": "e491eda904a7504f8c8fb6ccb9011be947a325e2072c9c5e9bfeecffc284420c",
}
COLORS = {"validation": "#0072B2", "test": "#D55E00", "correct": "#56B4E9", "taxonomy": "#009E73", "first": "#E69F00"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_member(zf: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    names = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise AssertionError(f"expected one {suffix} in {zf.filename}: {names}")
    return json.loads(zf.read(names[0]))


def top_level_metadata(zf: zipfile.ZipFile) -> dict[str, Any]:
    """Select run metadata, excluding the evaluator's archived copy."""
    names = [
        name for name in zf.namelist()
        if name.endswith("/metadata.json") and "/results/" not in name
    ]
    if len(names) != 1:
        raise AssertionError(f"expected one top-level metadata.json in {zf.filename}: {names}")
    return json.loads(zf.read(names[0]))


def jsonl_member(zf: zipfile.ZipFile, suffix: str) -> list[dict[str, Any]]:
    names = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise AssertionError(f"expected one {suffix} in {zf.filename}: {names}")
    return [json.loads(line) for line in zf.read(names[0]).decode().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label_f1(records: list[dict[str, Any]], label: str) -> tuple[float, int]:
    valid = [row for row in records if row["output_valid"]]
    pairs = [(row["gold_error_taxonomy"], row["parsed_output"]["error_taxonomy"]) for row in valid]
    tp = sum(gold == label and pred == label for gold, pred in pairs)
    fp = sum(gold != label and pred == label for gold, pred in pairs)
    fn = sum(gold == label and pred != label for gold, pred in pairs)
    denom = 2 * tp + fp + fn
    return (0 if denom == 0 else 2 * tp / denom), sum(gold == label for gold, _ in pairs)


def slice_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dimension in ("topic", "difficulty"):
        for value in sorted({row[dimension] for row in records}):
            group = [row for row in records if row[dimension] == value]
            valid = [row for row in group if row["output_valid"]]
            localized = [row for row in valid if isinstance(row["gold_first_error_step"], int)]
            rows.append({
                "dimension": dimension, "slice": value, "examples": len(group), "valid": len(valid),
                "correct_exact": sum(bool(row.get("correctness_exact")) for row in valid),
                "correct_rate": sum(bool(row.get("correctness_exact")) for row in valid) / len(valid) if valid else 0,
                "taxonomy_exact": sum(bool(row.get("taxonomy_exact")) for row in valid),
                "taxonomy_rate": sum(bool(row.get("taxonomy_exact")) for row in valid) / len(valid) if valid else 0,
                "first_error_support": len(localized),
                "first_error_exact": sum(bool(row.get("first_error_exact")) for row in localized),
                "first_error_rate": sum(bool(row.get("first_error_exact")) for row in localized) / len(localized) if localized else 0,
            })
    return rows


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    svg_path = FIG / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    # Matplotlib emits spaces before newlines in path data; normalize the
    # versioned SVG so git integrity checks remain signal-bearing.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def render_notebook() -> None:
    path = ROOT / "notebooks/v0.8-final-test-reproducibility.ipynb"
    if not path.exists():
        return
    notebook = json.loads(path.read_text())
    cells = []
    for cell in notebook["cells"]:
        source = html.escape("".join(cell.get("source", [])))
        if cell["cell_type"] == "markdown":
            cells.append(f'<section class="markdown"><pre>{source}</pre></section>')
        else:
            outputs = []
            for output in cell.get("outputs", []):
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                outputs.append(f'<pre class="output">{html.escape(str(text))}</pre>')
            cells.append(f'<section class="code"><div>In [{cell.get("execution_count") or " "}]</div><pre>{source}</pre>{"".join(outputs)}</section>')
    figures = "".join(f'<figure><img src="figures/{p.name}"><figcaption>{p.stem}</figcaption></figure>' for p in sorted(FIG.glob("*.png")))
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>MURL v0.8 final-test reproduction</title><style>
body{{font:16px system-ui;max-width:1100px;margin:2rem auto}}section{{border:1px solid #ddd;border-radius:8px;margin:1rem 0;padding:1rem}}pre{{white-space:pre-wrap;font:14px ui-monospace}}.code{{background:#f7f9fb}}.output{{background:white;border-left:4px solid #0072b2;padding:.7rem}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}img{{width:100%;border:1px solid #ddd}}figure{{margin:0}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><h1>MURL v0.8 final-test reproducibility notebook</h1>{''.join(cells)}<h2>Figures</h2><div class="grid">{figures}</div></body></html>'''
    (OUT / "notebook-render.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        render_notebook()
        return 0
    for key, path in (("final_zip", FINAL_ZIP), ("raw_zip", RAW_ZIP), ("executed_notebook", EXECUTED_NOTEBOOK), ("validation_zip", VALIDATION_ZIP)):
        if sha256(path) != EXPECTED[key]:
            raise AssertionError(f"{key} hash mismatch")
    with zipfile.ZipFile(FINAL_ZIP) as zf:
        assert zf.testzip() is None
        aggregate = json_member(zf, "/results/aggregate.json")
        test_records = jsonl_member(zf, "/results/per-example.jsonl")
        metadata = top_level_metadata(zf)
        manifest = json_member(zf, "/sha256-manifest.json")
    with zipfile.ZipFile(RAW_ZIP) as zf:
        assert zf.testzip() is None
    with zipfile.ZipFile(VALIDATION_ZIP) as zf:
        validation_aggregate = json_member(zf, "/results/aggregate.json")
        validation_records = jsonl_member(zf, "/results/per-example.jsonl")
    if aggregate["examples"] != 47 or aggregate["valid_outputs"] != 47 or len(test_records) != 47:
        raise AssertionError("final-test count mismatch")
    if manifest["generation_passes"] != 1 or manifest["scoring_invocations"] != 1:
        raise AssertionError("one-time execution invariant failed")
    correct_test = sum(bool(row.get("correctness_exact")) for row in test_records)
    correct_validation = sum(bool(row.get("correctness_exact")) for row in validation_records)
    summary = [
        {"split": "validation", "examples": 48, "valid": validation_aggregate["valid_outputs"], "correct": correct_validation,
         "validity_rate": validation_aggregate["output_validity_rate"], "correct_rate_all": correct_validation / 48,
         "correct_accuracy_valid_only": validation_aggregate["correctness_accuracy_valid_only"],
         "correct_macro_f1_valid_only": validation_aggregate["correctness_macro_f1_valid_only"],
         "taxonomy_macro_f1": validation_aggregate["taxonomy_macro_f1_valid_only"],
         "first_error_support": validation_aggregate["first_error_eligible_examples"], "first_error_exact": validation_aggregate["first_error_exact_match_valid_only"],
         "first_error_mad": validation_aggregate["first_error_mean_absolute_distance_valid_only"]},
        {"split": "test", "examples": 47, "valid": aggregate["valid_outputs"], "correct": correct_test,
         "validity_rate": aggregate["output_validity_rate"], "correct_rate_all": correct_test / 47,
         "correct_accuracy_valid_only": aggregate["correctness_accuracy_valid_only"],
         "correct_macro_f1_valid_only": aggregate["correctness_macro_f1_valid_only"],
         "taxonomy_macro_f1": aggregate["taxonomy_macro_f1_valid_only"],
         "first_error_support": aggregate["first_error_eligible_examples"], "first_error_exact": aggregate["first_error_exact_match_valid_only"],
         "first_error_mad": aggregate["first_error_mean_absolute_distance_valid_only"]},
    ]
    write_csv(OUT / "summary.csv", summary)
    slices = slice_table(test_records)
    write_csv(OUT / "test-slices.csv", slices)
    labels = sorted({row["gold_error_taxonomy"] for row in test_records} | {row["gold_error_taxonomy"] for row in validation_records})
    label_rows = []
    for label in labels:
        test_f1, test_n = label_f1(test_records, label)
        val_f1, val_n = label_f1(validation_records, label)
        label_rows.append({"label": label, "validation_support": val_n, "validation_f1": val_f1, "test_support": test_n, "test_f1": test_f1})
    write_csv(OUT / "per-label-f1.csv", label_rows)
    valid_test = [row for row in test_records if row["output_valid"]]
    confusion = Counter((row["gold_error_taxonomy"], row["parsed_output"]["error_taxonomy"]) for row in valid_test)
    confusion_rows = [{"gold": g, "predicted": p, "count": confusion[(g, p)]} for g in labels for p in labels]
    write_csv(OUT / "taxonomy-confusion.csv", confusion_rows)
    artifact_manifest = {
        "run_id": RUN_ID, "generation_passes": 1, "scoring_invocations": 1,
        "final_zip": {"path": str(FINAL_ZIP.relative_to(ROOT)), "sha256": EXPECTED["final_zip"]},
        "raw_pre_score_zip": {"path": str(RAW_ZIP.relative_to(ROOT)), "sha256": EXPECTED["raw_zip"]},
        "executed_colab_notebook": {"path": str(EXECUTED_NOTEBOOK.relative_to(ROOT)), "sha256": EXPECTED["executed_notebook"]},
        "validation_zip": {"path": str(VALIDATION_ZIP.relative_to(ROOT)), "sha256": EXPECTED["validation_zip"]},
        "frozen_input_sha256": metadata["input_sha256"], "setup_failures": metadata.get("setup_failures", []),
    }
    (OUT / "artifact-manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.titlesize": 14})
    metrics = [("Validity", lambda r: r["validity_rate"]), ("Correctness", lambda r: r["correct_accuracy_valid_only"]),
               ("Taxonomy macro-F1", lambda r: r["taxonomy_macro_f1"]), ("First-error exact", lambda r: r["first_error_exact"])]
    fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(len(metrics)); width = .34
    for offset, row in ((-.5, summary[0]), (.5, summary[1])):
        values = [getter(row) for _, getter in metrics]
        bars = ax.bar(x + offset * width, values, width, label=row["split"].title(), color=COLORS[row["split"]])
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_xticks(x, [name for name, _ in metrics]); ax.set_ylabel("Proportion / score")
    ax.set_title("Frozen Qwen3 v0.6: validation selection versus one-time held-out test")
    ax.grid(axis="y", alpha=.2); ax.legend();
    ax.text(.5, -.16, "Validation n=48; test n=47. First-error supports: 28 and 26. Axes start at zero.", transform=ax.transAxes, ha="center")
    save(fig, "validation-vs-test")

    fig, axes = plt.subplots(2, 1, figsize=(12, 11))
    for ax, dimension in zip(axes, ("topic", "difficulty")):
        rows = [row for row in slices if row["dimension"] == dimension]
        x = np.arange(len(rows)); width = .26
        for i, (field, name, color) in enumerate((("correct_rate", "Correctness", COLORS["correct"]), ("taxonomy_rate", "Taxonomy exact", COLORS["taxonomy"]), ("first_error_rate", "First-error exact", COLORS["first"]))):
            ax.bar(x + (i - 1) * width, [row[field] for row in rows], width, label=name, color=color)
        ax.set_ylim(0, 1); ax.set_ylabel("Exact-match rate"); ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8)
        ax.set_xticks(
            x,
            [f"{row['slice'].replace('_', ' ')}\n(n={row['examples']}; error n={row['first_error_support']})" for row in rows],
            rotation=12 if dimension == "topic" else 8,
            ha="right",
            fontsize=8,
        )
        ax.set_title(f"By {dimension}")
    fig.suptitle("One-time held-out test slices (descriptive; sparse supports)")
    fig.subplots_adjust(top=.92, bottom=.10, hspace=.58)
    save(fig, "test-topic-difficulty-slices")

    fig, ax = plt.subplots(figsize=(12, 6)); x = np.arange(len(label_rows)); width = .36
    ax.bar(x - width/2, [row["validation_f1"] for row in label_rows], width, label="Validation", color=COLORS["validation"])
    ax.bar(x + width/2, [row["test_f1"] for row in label_rows], width, label="Test", color=COLORS["test"])
    ax.set_xticks(x, [f"{row['label']}\n(v={row['validation_support']}, t={row['test_support']})" for row in label_rows], rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("One-vs-rest F1"); ax.grid(axis="y", alpha=.2); ax.legend()
    ax.set_title("Taxonomy performance by label: validation and held-out test")
    save(fig, "per-label-validation-vs-test")

    matrix = np.zeros((len(labels), len(labels)), dtype=int); positions = {label: i for i, label in enumerate(labels)}
    for row in confusion_rows:
        matrix[positions[row["gold"]], positions[row["predicted"]]] = row["count"]
    fig, ax = plt.subplots(figsize=(9, 8)); image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted taxonomy"); ax.set_ylabel("Gold taxonomy"); ax.set_title("Held-out taxonomy confusion (47/47 schema-valid)")
    fig.colorbar(image, ax=ax, shrink=.8, label="Count"); save(fig, "test-taxonomy-confusion")

    val_dist = Counter(row.get("first_error_distance") for row in validation_records if isinstance(row["gold_first_error_step"], int) and row["output_valid"])
    test_dist = Counter(row.get("first_error_distance") for row in test_records if isinstance(row["gold_first_error_step"], int) and row["output_valid"])
    distances = sorted({d for d in val_dist | test_dist if d is not None})
    categories = distances + [None]
    labels_distance = [str(distance) for distance in distances] + ["No step"]
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(categories)); width = .36
    ax.bar(x-width/2, [val_dist[d] for d in categories], width, label="Validation (eligible n=28)", color=COLORS["validation"])
    ax.bar(x+width/2, [test_dist[d] for d in categories], width, label="Test (eligible n=26)", color=COLORS["test"])
    ax.set_xticks(x, labels_distance); ax.set_xlabel("Absolute first-error-step distance"); ax.set_ylabel("Examples")
    ax.set_title("First-error localization distance"); ax.grid(axis="y", alpha=.2); ax.legend(); save(fig, "first-error-distance")
    render_notebook()
    print(json.dumps({"result": "PASS", "test_valid": 47, "test_correct": correct_test, "final_zip_sha256": EXPECTED["final_zip"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
