#!/usr/bin/env python3
"""Reproduce the MURL v0.8 cross-family audit from archived artifacts.

This script deliberately opens only archived *validation* and training bundles.
It never opens a held-out test split or a test-results archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/murl-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "baseline"
OUT = ROOT / "reports" / "v0.8-final"
FIG = OUT / "figures"

sys.path.insert(0, str(ROOT))
from evaluation.evaluate import evaluate  # noqa: E402

GATES = {"valid": 47, "correct": 30, "taxonomy": 0.3201, "first_error": 0.536}
COLORS = {"Qwen2.5": "#7A7A7A", "Qwen3": "#0072B2", "Phi": "#D55E00"}

VALIDATION_ARCHIVES = {
    "qwen3-base": "qwen3-4b-v08-base-control-validation-results.zip",
    "qwen3-v0.5": "qwen3-4b-v08-v05-validation-results.zip",
    "qwen3-v0.6": "qwen3-4b-v08-v06-validation-results.zip",
    "qwen3-v0.7": "qwen3-4b-v08-v07-validation-results.zip",
    "phi-base": "phi4-mini-v08-base-control-validation-results.zip",
    "phi-v0.5": "phi4-mini-v08-v05-validation-results.zip",
    "phi-v0.6": "phi4-mini-v08-v06-validation-results.zip",
    "phi-v0.7": "phi4-mini-v08-v07-validation-results.zip",
    "qwen2.5-v0.5": "qlora-v05-moderated-taxonomy-balance-validation-results.zip",
    "qwen2.5-v0.6": "qlora-v06-hierarchical-multitask-validation-results.zip",
    "qwen2.5-v0.7": "qlora-v07-two-stage-diagnosis-validation-results.zip",
}

TRAINING_ARCHIVES = [
    f"{family}-v08-v0{recipe}-training-results.zip"
    for family in ("qwen3-4b", "phi4-mini")
    for recipe in (5, 6, 7)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_member(zf: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    names = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise AssertionError(f"expected one {suffix!r} in {zf.filename}, got {names}")
    return json.loads(zf.read(names[0]))


def jsonl_member(zf: zipfile.ZipFile, suffix: str) -> list[dict[str, Any]]:
    names = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise AssertionError(f"expected one {suffix!r} in {zf.filename}, got {names}")
    return [json.loads(line) for line in zf.read(names[0]).decode().splitlines() if line.strip()]


def family_for(key: str) -> str:
    if key.startswith("qwen2.5"):
        return "Qwen2.5"
    if key.startswith("qwen3"):
        return "Qwen3"
    return "Phi"


def label_for(key: str) -> str:
    return key.replace("qwen2.5", "Qwen2.5").replace("qwen3", "Qwen3").replace("phi", "Phi")


def audit_training_archives() -> list[dict[str, Any]]:
    rows = []
    required_suffixes = [
        "/metadata.json", "/resolved_config.json", "/dataset_metadata.json",
        "/training_metrics.json", "/adapter/adapter_config.json",
        "/adapter/adapter_model.safetensors", "/tokenizer/tokenizer.json",
    ]
    for filename in TRAINING_ARCHIVES:
        path = RUNS / filename
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            missing = [suffix for suffix in required_suffixes if not any(n.endswith(suffix) for n in names)]
            if missing:
                raise AssertionError(f"{filename}: missing {missing}")
            if not any(n.endswith("/trainer_state.json") for n in names):
                raise AssertionError(f"{filename}: missing trainer state")
            metadata = json_member(zf, "/metadata.json")
            config = json_member(zf, "/resolved_config.json")
            dataset_metadata = json_member(zf, "/dataset_metadata.json")
            training = config["training"]
            if training["max_steps"] != 60 or training["max_sequence_length"] != 2048:
                raise AssertionError(f"{filename}: frozen recipe mismatch")
            if config["run"]["seed"] != 20261004:
                raise AssertionError(f"{filename}: seed mismatch")
            if dataset_metadata["output_sha256"] != metadata["dataset_sha256"]:
                raise AssertionError(f"{filename}: dataset hash mismatch")
            final_states = [n for n in names if n.endswith("/checkpoint-60/trainer_state.json")]
            if len(final_states) != 1 or json.loads(zf.read(final_states[0]))["global_step"] != 60:
                raise AssertionError(f"{filename}: final trainer state mismatch")
            source_members = [n for n in names if n.endswith("/source.zip")]
            expected_source = (
                "murl-v08-qwen3-phi-source.zip" if "v05" in filename
                else "murl-v08-v06-cross-family-source.zip" if "v06" in filename
                else "murl-v08-v07-cross-family-source.zip"
            )
            embedded_source_hash = hashlib.sha256(zf.read(source_members[0])).hexdigest()
            if embedded_source_hash != sha256(RUNS / expected_source):
                raise AssertionError(f"{filename}: embedded source snapshot mismatch")
            rows.append({
                "archive": filename,
                "run_id": metadata["run_id"],
                "base_model": metadata["base_model"],
                "base_revision": metadata["base_revision"],
                "gpu": metadata["gpu"],
                "torch": metadata["torch"],
                "transformers": metadata["transformers"],
                "trainable_parameters": metadata["trainable_parameters"],
                "train_runtime_seconds": metadata["training_metrics"]["train_runtime"],
                "dataset_sha256": metadata["dataset_sha256"],
                "member_count": len(names),
                "embedded_source_sha256": embedded_source_hash,
                "final_global_step": 60,
                "required_contents": "PASS",
            })
    return rows


def audit_validation_archives() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    per_examples: dict[str, list[dict[str, Any]]] = {}
    for key, filename in VALIDATION_ARCHIVES.items():
        path = RUNS / filename
        with zipfile.ZipFile(path) as zf:
            required = ["/gold.jsonl", "/outputs.jsonl", "/metadata.json", "/prompt.md",
                        "/results/aggregate.json", "/results/per-example.jsonl"]
            missing = [suffix for suffix in required if not any(n.endswith(suffix) for n in zf.namelist())]
            if missing:
                raise AssertionError(f"{filename}: missing {missing}")
            gold = jsonl_member(zf, "/gold.jsonl")
            outputs = jsonl_member(zf, "/outputs.jsonl")
            if len(gold) != 48 or len(outputs) != 48:
                raise AssertionError(f"{filename}: expected 48 gold and output rows")
            metadata_names = [n for n in zf.namelist() if n.endswith("/metadata.json") and "/results/" not in n]
            if len(metadata_names) != 1:
                raise AssertionError(f"{filename}: missing unique run metadata")
            run_metadata = json.loads(zf.read(metadata_names[0]))
            if run_metadata.get("split") != "validation" or run_metadata.get("evaluated_examples") != 48:
                raise AssertionError(f"{filename}: validation metadata mismatch")
            if key.startswith("qwen3") and run_metadata["runtime"]["model_revision"] != "1cfa9a7208912126459214e8b04321603b3df60c":
                raise AssertionError(f"{filename}: Qwen revision mismatch")
            if key.startswith("phi") and run_metadata["runtime"]["model_revision"] != "cfbefacb99257ffa30c83adab238a50856ac3083":
                raise AssertionError(f"{filename}: Phi revision mismatch")
            archived = json_member(zf, "/results/aggregate.json")
            recomputed_per_example, recomputed = evaluate(gold, outputs)
            for metric in (
                "examples", "valid_outputs", "correctness_accuracy_valid_only",
                "taxonomy_macro_f1_valid_only", "first_error_eligible_examples",
                "first_error_exact_match_valid_only", "first_error_mean_absolute_distance_valid_only",
            ):
                a, b = archived[metric], recomputed[metric]
                if a is None or b is None:
                    if a != b:
                        raise AssertionError(f"{filename}: {metric} mismatch")
                elif not math.isclose(float(a), float(b), abs_tol=1e-12):
                    raise AssertionError(f"{filename}: {metric} mismatch: {a} != {b}")
            correct = sum(bool(row.get("correctness_exact")) for row in recomputed_per_example)
            utility = (correct / 48 + recomputed["taxonomy_macro_f1_valid_only"]
                       + recomputed["first_error_exact_match_valid_only"]) / 3
            # The precommit records the anchor-derived floors at four decimals
            # (taxonomy) and three decimals (first error). Apply those stated
            # precisions uniformly, so the anchor itself remains inclusive.
            eligible = (
                recomputed["valid_outputs"] >= GATES["valid"] and correct >= GATES["correct"]
                and round(recomputed["taxonomy_macro_f1_valid_only"], 4) >= GATES["taxonomy"]
                and round(recomputed["first_error_exact_match_valid_only"], 3) >= GATES["first_error"]
            )
            rows.append({
                "key": key,
                "family": family_for(key),
                "arm": key.split("-", 1)[1],
                "archive": filename,
                "examples": 48,
                "valid": recomputed["valid_outputs"],
                "correct_all_48": correct,
                "correctness_all_48": correct / 48,
                "correctness_valid_only": recomputed["correctness_accuracy_valid_only"],
                "taxonomy_macro_f1_valid_only": recomputed["taxonomy_macro_f1_valid_only"],
                "first_error_n": recomputed["first_error_eligible_examples"],
                "first_error_exact_valid_only": recomputed["first_error_exact_match_valid_only"],
                "first_error_mad_valid_only": recomputed["first_error_mean_absolute_distance_valid_only"],
                "utility": utility,
                "eligible": eligible,
                "aggregate_recalculation": "PASS",
            })
            per_examples[key] = recomputed_per_example
    return rows, per_examples


def archive_manifest(full_integrity: bool) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RUNS.glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            bad_member = zf.testzip() if full_integrity else None
            rows.append({
                "archive": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "members": len(zf.infolist()),
                "crc_test": ("PASS" if bad_member is None else f"FAIL:{bad_member}") if full_integrity else "NOT_CHECKED",
                "semantic_scope": "not opened" if "test" in path.name else "audited where in v0.8 scope",
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f1_for_label(records: list[dict[str, Any]], label: str) -> tuple[float, int]:
    valid = [r for r in records if r["output_valid"]]
    gold = [r["gold_error_taxonomy"] for r in valid]
    pred = [r["parsed_output"]["error_taxonomy"] for r in valid]
    tp = sum(g == label and p == label for g, p in zip(gold, pred))
    fp = sum(g != label and p == label for g, p in zip(gold, pred))
    fn = sum(g == label and p != label for g, p in zip(gold, pred))
    denom = 2 * tp + fp + fn
    return (0.0 if denom == 0 else 2 * tp / denom), sum(g == label for g in gold)


def make_analysis_tables(summary: list[dict[str, Any]], per: dict[str, list[dict[str, Any]]]) -> None:
    lookup = {row["key"]: row for row in summary}
    deltas = []
    for family, prefix in (("Qwen3", "qwen3"), ("Phi", "phi")):
        for newer, older in (("v0.6", "v0.5"), ("v0.7", "v0.5"), ("v0.7", "v0.6")):
            n, o = lookup[f"{prefix}-{newer}"], lookup[f"{prefix}-{older}"]
            deltas.append({
                "family": family, "contrast": f"{newer} - {older}",
                "valid_count": n["valid"] - o["valid"],
                "correct_count_all_48": n["correct_all_48"] - o["correct_all_48"],
                "taxonomy_f1": n["taxonomy_macro_f1_valid_only"] - o["taxonomy_macro_f1_valid_only"],
                "first_error_exact": n["first_error_exact_valid_only"] - o["first_error_exact_valid_only"],
                "first_error_mad": (n["first_error_mad_valid_only"] or 0) - (o["first_error_mad_valid_only"] or 0),
                "utility": n["utility"] - o["utility"],
            })
    write_csv(OUT / "recipe-deltas.csv", deltas)

    labels = sorted({r["gold_error_taxonomy"] for r in per["qwen3-v0.6"]})
    label_rows = []
    for label in labels:
        row: dict[str, Any] = {"label": label}
        for key in ("qwen3-v0.5", "qwen3-v0.6", "qwen3-v0.7"):
            score, support = f1_for_label(per[key], label)
            row[f"{key}_f1"] = score
            row[f"{key}_support"] = support
        label_rows.append(row)
    write_csv(OUT / "qwen3-per-label-f1.csv", label_rows)

    records = [r for r in per["qwen3-v0.6"] if r["output_valid"]]
    confusion = Counter((r["gold_error_taxonomy"], r["parsed_output"]["error_taxonomy"]) for r in records)
    confusion_rows = [
        {"gold": gold, "predicted": pred, "count": confusion[(gold, pred)]}
        for gold in labels for pred in labels
    ]
    write_csv(OUT / "qwen3-v0.6-taxonomy-confusion.csv", confusion_rows)

    slice_rows = []
    for dimension in ("topic", "difficulty"):
        values = sorted({r[dimension] for r in per["qwen3-v0.6"]})
        for value in values:
            subset = [r for r in per["qwen3-v0.6"] if r[dimension] == value]
            valid = [r for r in subset if r["output_valid"]]
            localized = [r for r in valid if isinstance(r["gold_first_error_step"], int)]
            slice_rows.append({
                "dimension": dimension, "slice": value, "examples": len(subset), "valid": len(valid),
                "correctness": sum(bool(r.get("correctness_exact")) for r in valid) / len(valid) if valid else 0,
                "taxonomy_exact": sum(bool(r.get("taxonomy_exact")) for r in valid) / len(valid) if valid else 0,
                "first_error_exact": sum(bool(r.get("first_error_exact")) for r in localized) / len(localized) if localized else 0,
                "first_error_support": len(localized),
            })
    write_csv(OUT / "qwen3-v0.6-slices.csv", slice_rows)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_notebook_html() -> Path:
    """Create a dependency-free static inspection render of the notebook."""
    notebook_path = ROOT / "notebooks" / "v0.8-final-reproducibility.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        source = html.escape("".join(cell.get("source", [])))
        if cell["cell_type"] == "markdown":
            cells.append(f'<section class="markdown"><pre>{source}</pre></section>')
            continue
        count = cell.get("execution_count")
        tag = "disabled" if "disabled" in cell.get("metadata", {}).get("tags", []) else "code"
        outputs = []
        for output in cell.get("outputs", []):
            text = output.get("text", output.get("data", {}).get("text/plain", ""))
            if isinstance(text, list):
                text = "".join(text)
            outputs.append(f'<pre class="output">{html.escape(str(text))}</pre>')
        cells.append(
            f'<section class="{tag}"><div class="prompt">In [{count if count is not None else " "}]</div>'
            f'<pre>{source}</pre>{"".join(outputs)}</section>'
        )
    figures = "".join(
        f'<figure><img src="figures/{path.name}" alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
        for path in sorted(FIG.glob("*.png"))
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>MURL v0.8 reproducibility notebook</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;color:#202124}}section{{border:1px solid #ddd;border-radius:8px;margin:1rem 0;padding:1rem}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:.4rem 0;font:14px ui-monospace,monospace}}.markdown{{background:#fff}}.code{{background:#f7f9fb}}.disabled{{background:#fff3cd;border-color:#d9a400}}.prompt{{color:#666;font:13px ui-monospace}}.output{{background:white;border-left:4px solid #0072b2;padding:.75rem}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}}figure{{margin:0}}img{{width:100%;height:auto;border:1px solid #ddd}}figcaption{{text-align:center;color:#555}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>MURL v0.8 final reproducibility notebook — static render</h1>{''.join(cells)}<h2>Generated figures</h2><div class="grid">{figures}</div></body></html>"""
    output = OUT / "notebook-render.html"
    output.write_text(document, encoding="utf-8")
    return output


def plot_all(summary: list[dict[str, Any]]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10, "figure.titlesize": 14})
    lookup = {row["key"]: row for row in summary}
    order = ["qwen2.5-v0.5", "qwen2.5-v0.6", "qwen2.5-v0.7", "qwen3-base", "qwen3-v0.5",
             "qwen3-v0.6", "qwen3-v0.7", "phi-base", "phi-v0.5", "phi-v0.6", "phi-v0.7"]
    metrics = [
        ("valid", "Schema validity", lambda r: r["valid"] / 48),
        ("correct", "Correctness (all 48)", lambda r: r["correctness_all_48"]),
        ("taxonomy", "Taxonomy macro-F1", lambda r: r["taxonomy_macro_f1_valid_only"]),
        ("first", "First-error exact", lambda r: r["first_error_exact_valid_only"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    y = np.arange(len(order))
    for ax, (_, title, getter) in zip(axes.flat, metrics):
        values = [getter(lookup[k]) for k in order]
        colors = [COLORS[lookup[k]["family"]] for k in order]
        ax.barh(y, values, color=colors, alpha=.88)
        ax.set_xlim(0, 1); ax.set_title(title); ax.grid(axis="x", alpha=.2)
        ax.set_yticks(y, [label_for(k) for k in order]); ax.invert_yaxis()
        for yi, value in zip(y, values):
            ax.text(min(value + .012, .96), yi, f"{value:.2f}", va="center", fontsize=7)
    fig.suptitle("MURL validation comparison: archived and independently recalculated")
    fig.text(.5, .01, "Classification/localization metrics are schema-valid-only except correctness, which uses all 48. Axes start at zero.", ha="center")
    save_figure(fig, "main-comparison")

    candidates = ["qwen2.5-v0.5", "qwen2.5-v0.6", "qwen2.5-v0.7", "qwen3-v0.5", "qwen3-v0.6",
                  "qwen3-v0.7", "phi-v0.5", "phi-v0.6", "phi-v0.7"]
    gate_values = [
        ("Validity", lambda r: r["valid"] / GATES["valid"]),
        ("Correctness", lambda r: r["correct_all_48"] / GATES["correct"]),
        ("Taxonomy F1", lambda r: r["taxonomy_macro_f1_valid_only"] / GATES["taxonomy"]),
        ("First-error", lambda r: r["first_error_exact_valid_only"] / GATES["first_error"]),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(candidates)); width = .19
    palette = ["#56B4E9", "#009E73", "#E69F00", "#CC79A7"]
    for i, (name, getter) in enumerate(gate_values):
        vals = [getter(lookup[k]) for k in candidates]
        ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=palette[i])
    ax.axhline(1, color="black", lw=1.2, ls="--", label="Eligibility threshold")
    ax.set_ylim(0, 2.3); ax.set_ylabel("Result / precommitted threshold")
    ax.set_xticks(x, [label_for(k) for k in candidates], rotation=32, ha="right")
    ax.set_title("Eligibility gate audit (all four bars must reach 1.0)")
    ax.grid(axis="y", alpha=.2); ax.legend(ncol=5, fontsize=8, loc="upper left")
    save_figure(fig, "eligibility-gates")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    contrasts = [("v0.6 − v0.5", "v0.6", "v0.5"), ("v0.7 − v0.5", "v0.7", "v0.5"), ("v0.7 − v0.6", "v0.7", "v0.6")]
    fields = [("Correct / 48", lambda r: r["correctness_all_48"]), ("Taxonomy F1", lambda r: r["taxonomy_macro_f1_valid_only"]),
              ("First-error", lambda r: r["first_error_exact_valid_only"]), ("Utility", lambda r: r["utility"])]
    for ax, (family, prefix) in zip(axes, (("Qwen3", "qwen3"), ("Phi", "phi"))):
        x = np.arange(len(contrasts)); width = .2
        for i, (name, getter) in enumerate(fields):
            vals = [getter(lookup[f"{prefix}-{new}"]) - getter(lookup[f"{prefix}-{old}"]) for _, new, old in contrasts]
            ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=palette[i])
        ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, [c[0] for c in contrasts], rotation=20)
        ax.set_title(f"{family} within-family recipe deltas"); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Absolute change (proportion)"); axes[1].legend(fontsize=8)
    save_figure(fig, "recipe-deltas")

    categories = ["Validity", "Correct / 48", "Taxonomy F1", "First-error"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for arm, color in (("base", "#999999"), ("v0.5", "#56B4E9"), ("v0.6", "#0072B2"), ("v0.7", "#009E73")):
        r = lookup[f"qwen3-{arm}"]
        vals = [r["valid"] / 48, r["correctness_all_48"], r["taxonomy_macro_f1_valid_only"], r["first_error_exact_valid_only"]]
        vals += vals[:1]; ax.plot(angles, vals, lw=2, label=arm, color=color); ax.fill(angles, vals, alpha=.05, color=color)
    ax.set_xticks(angles[:-1], categories); ax.set_ylim(0, 1); ax.set_yticks([.25, .5, .75, 1])
    ax.set_title("Qwen3 profile: v0.6 is the most balanced eligible arm", pad=25); ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    save_figure(fig, "qwen3-profile")

    fig, ax = plt.subplots(figsize=(10, 5))
    phi_order = ["phi-base", "phi-v0.5", "phi-v0.6", "phi-v0.7"]
    x = np.arange(4); width = .2
    for i, (_, name, getter) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [getter(lookup[k]) for k in phi_order], width, label=name, color=palette[i])
    ax.set_ylim(0, 1); ax.set_xticks(x, [label_for(k) for k in phi_order]); ax.grid(axis="y", alpha=.2)
    ax.set_title("Phi adaptation repairs validity, but no trained arm clears every gate")
    ax.set_ylabel("Proportion / score"); ax.legend(ncol=2, fontsize=8)
    ax.text(0, .05, "Base classification metrics use only 4 valid outputs", ha="center", fontsize=8, rotation=90)
    save_figure(fig, "phi-improvement")

    label_rows = list(csv.DictReader((OUT / "qwen3-per-label-f1.csv").open()))
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(label_rows)); width = .26
    for i, (arm, color) in enumerate((("v0.5", "#56B4E9"), ("v0.6", "#0072B2"), ("v0.7", "#009E73"))):
        vals = [float(r[f"qwen3-{arm}_f1"]) for r in label_rows]
        ax.bar(x + (i - 1) * width, vals, width, label=arm, color=color)
    supports = [int(r["qwen3-v0.6_support"]) for r in label_rows]
    ax.set_xticks(x, [f"{r['label']}\n(n={n})" for r, n in zip(label_rows, supports)], rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("One-vs-rest F1"); ax.grid(axis="y", alpha=.2); ax.legend()
    ax.set_title("Qwen3 taxonomy performance by label (tiny supports require caution)")
    save_figure(fig, "qwen3-per-label")

    slice_rows = list(csv.DictReader((OUT / "qwen3-v0.6-slices.csv").open()))
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    for ax, dimension in zip(axes, ("topic", "difficulty")):
        selected = [r for r in slice_rows if r["dimension"] == dimension]
        x = np.arange(len(selected)); width = .26
        for i, (field, name, color) in enumerate((("correctness", "Correctness", "#56B4E9"), ("taxonomy_exact", "Taxonomy exact", "#009E73"), ("first_error_exact", "First-error exact", "#E69F00"))):
            ax.bar(x + (i - 1) * width, [float(r[field]) for r in selected], width, label=name, color=color)
        ax.set_xticks(x, [f"{r['slice'].replace('_', ' ')}\n(n={r['examples']})" for r in selected], rotation=20, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel("Exact-match rate"); ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8)
        ax.set_title(f"By {dimension}")
    fig.suptitle("Qwen3 v0.6 validation slices (small supports; descriptive only)")
    save_figure(fig, "qwen3-topic-difficulty-slices")

    confusion_rows = list(csv.DictReader((OUT / "qwen3-v0.6-taxonomy-confusion.csv").open()))
    labels = sorted({r["gold"] for r in confusion_rows})
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    positions = {label: i for i, label in enumerate(labels)}
    for row in confusion_rows:
        matrix[positions[row["gold"]], positions[row["predicted"]]] = int(row["count"])
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("Predicted taxonomy"); ax.set_ylabel("Gold taxonomy")
    ax.set_title("Qwen3 v0.6 taxonomy confusion (47 schema-valid outputs)")
    fig.colorbar(image, ax=ax, shrink=.8, label="Count")
    save_figure(fig, "qwen3-v0.6-confusion")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-full-integrity", action="store_true", help="Skip CRC reads for speed; hashes are still calculated.")
    parser.add_argument("--render-only", action="store_true", help="Only refresh the static notebook HTML render.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        print(render_notebook_html())
        return 0
    manifest = archive_manifest(not args.skip_full_integrity)
    if any(row["crc_test"].startswith("FAIL") for row in manifest):
        raise AssertionError("one or more ZIP CRC checks failed")
    training = audit_training_archives()
    summary, per = audit_validation_archives()
    write_csv(OUT / "archive-manifest.csv", manifest)
    write_csv(OUT / "training-artifact-audit.csv", training)
    write_csv(OUT / "model-summary.csv", summary)
    make_analysis_tables(summary, per)
    plot_all(summary)
    render_notebook_html()
    decision = {
        "gates": GATES,
        "family_nominees": {"Qwen2.5": "qwen2.5-v0.5", "Qwen3": "qwen3-v0.6", "Phi": None},
        "winner": "qwen3-v0.6",
        "held_out_test_executed": False,
        "validation_archives_recalculated": len(summary),
        "training_archives_audited": len(training),
        "zip_archives_hashed": len(manifest),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
