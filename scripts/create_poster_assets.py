#!/usr/bin/env python3
"""Create deterministic, data-backed MURL poster visuals."""

from __future__ import annotations

import csv
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "poster-assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#12304A"
BLUE = "#1F77B4"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
ORANGE = "#F4A261"
RED = "#D95D39"
GRAY = "#607080"
PALE = "#F5F8FA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
plt.rcParams["svg.hashsalt"] = "murl-poster-assets"


def save(fig: plt.Figure, stem: str, *, tight: bool = True) -> None:
    bbox_inches = "tight" if tight else None
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches=bbox_inches)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches=bbox_inches, metadata={"Date": None})
    plt.close(fig)


def make_methods_figure() -> None:
    rows = [json.loads(line) for line in (ROOT / "data/raw/graded-exams.jsonl").read_text().splitlines()]
    methods = [
        ("substitution", "Substitution", "u = g(x)"),
        ("integration_by_parts", "Integration by parts", "∫u dv = uv − ∫v du"),
        ("trig_integrals", "Trigonometric integrals", "identities + saved factor"),
        ("trig_substitution", "Trigonometric substitution", "x = a sin θ, a tan θ, or a sec θ"),
        ("partial_fractions", "Partial fractions", "decompose rational functions"),
        ("definite_integrals", "Definite integrals / FTC", "F(b) − F(a)"),
        ("improper_integrals", "Improper integrals", "replace endpoint with a limit"),
    ]
    labels = [m[1] for m in methods]
    cues = [m[2] for m in methods]
    classes = ["correct", "minor_error", "incorrect", "insufficient_work"]
    class_labels = ["Correct", "Minor error", "Incorrect", "Insufficient work"]
    colors = [TEAL, GOLD, RED, GRAY]
    counts = []
    for key, _, _ in methods:
        c = Counter(r["overall_correctness"] for r in rows if r["topic"] == key)
        counts.append([c[k] for k in classes])
    counts = np.asarray(counts)
    totals = counts.sum(axis=1)

    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for j, (name, color) in enumerate(zip(class_labels, colors)):
        bars = ax.barh(y, counts[:, j], left=left, height=0.64, color=color, label=name, edgecolor="white")
        for i, bar in enumerate(bars):
            value = int(counts[i, j])
            if value:
                ax.text(left[i] + value / 2, i, str(value), ha="center", va="center", fontsize=10, fontweight="bold", color="white" if j in (0, 2, 3) else NAVY)
        left += counts[:, j]

    for i, (total, cue) in enumerate(zip(totals, cues)):
        ax.text(total + 0.7, i - 0.10, f"n = {total}", va="center", fontsize=11, fontweight="bold")
        ax.text(total + 0.7, i + 0.18, cue, va="center", fontsize=9.5, color=GRAY)

    ax.set_yticks(y, labels, fontsize=11.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) + 18)
    ax.set_xlabel("Number of authored benchmark examples", fontsize=11.5, fontweight="bold")
    ax.set_title("Seven Core Integration Methods in the MURL Benchmark", loc="left", fontsize=22, pad=24, color=NAVY)
    ax.text(0, 1.035, "Actual composition of 203 method-specific records; the separate mixed-review category contains 47 additional records.", transform=ax.transAxes, fontsize=11.5, color=GRAY)
    ax.grid(axis="x", color="#D7E0E6", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, -0.18), frameon=False, fontsize=10.5)
    ax.text(1, -0.16, "Source: data/raw/graded-exams.jsonl (250 records total)", transform=ax.transAxes, ha="right", fontsize=9.5, color=GRAY)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.84, bottom=0.20)
    save(fig, "seven-integration-methods")


def make_cross_model_figure() -> None:
    with (ROOT / "reports/v0.8-final/model-summary.csv").open(newline="") as f:
        raw = list(csv.DictReader(f))
    family_order = ["Qwen2.5", "Qwen3", "Phi"]
    arm_order = {"base": 0, "v0.5": 1, "v0.6": 2, "v0.7": 3}
    raw.sort(key=lambda r: (family_order.index(r["family"]), arm_order[r["arm"]]))

    metrics = [
        ("Validity", lambda r: int(r["valid"]) / int(r["examples"])),
        ("Correctness\n(all 48)", lambda r: float(r["correctness_all_48"])),
        ("Taxonomy\nmacro-F1", lambda r: float(r["taxonomy_macro_f1_valid_only"])),
        ("First-error\nexact", lambda r: float(r["first_error_exact_valid_only"])),
        ("Combined\nutility", lambda r: float(r["utility"])),
    ]
    values = np.array([[fn(r) for _, fn in metrics] for r in raw])
    labels = [f"{r['family']}  {r['arm']}" for r in raw]
    winner_index = next(i for i, r in enumerate(raw) if r["key"] == "qwen3-v0.6")

    cmap = LinearSegmentedColormap.from_list("murl", ["#EEF4F7", "#9CC9D3", "#2A9D8F", "#176B78"])
    fig, ax = plt.subplots(figsize=(13.8, 9.0))
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            color = "white" if v >= 0.64 else NAVY
            text = f"{v:.3f}"
            if j == 0:
                text += f"\n({raw[i]['valid']}/48)"
            elif j == 1:
                text += f"\n({raw[i]['correct_all_48']}/48)"
            elif j == 3:
                text += f"\n({round(v * int(raw[i]['first_error_n']))}/{raw[i]['first_error_n']})"
            ax.text(j, i, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)

    ax.set_xticks(range(len(metrics)), [m[0] for m in metrics], fontsize=11, fontweight="bold")
    display_labels = []
    for i, label in enumerate(labels):
        display_labels.append(("★  " if i == winner_index else "") + label)
    ax.set_yticks(range(len(raw)), display_labels, fontsize=10.5)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=10)

    for boundary in (2.5, 6.5):
        ax.axhline(boundary, color="white", linewidth=16)
        ax.axhline(boundary, color=NAVY, linewidth=3.0, alpha=0.65)
    ax.add_patch(Rectangle((-0.5, winner_index - 0.5), len(metrics), 1, fill=False, edgecolor="#F4B942", linewidth=4))

    # Center over the complete visual composition (row labels through colorbar),
    # rather than the raw canvas, whose asymmetric margins make 0.5 look left-shifted.
    fig.suptitle("Cross-Family Validation", x=0.545, y=0.965, ha="center", fontsize=24, fontweight="bold", color=NAVY)
    footer_lines = [
        "• Darker cells = stronger results.",
        "• Validity = usable JSON.  Correctness = the right grade.",
        "• Taxonomy = the right error type.  First-error = the right first wrong step.",
        "• Every model used the same 48 validation problems.",
        "• ★ = selected model: it met every minimum requirement and earned the highest combined utility.",
    ]
    for line_y, line in zip((0.175, 0.145, 0.115, 0.085, 0.055), footer_lines):
        fig.text(0.23, line_y, line, fontsize=9.5, color=GRAY)
    fig.text(0.92, 0.022, "Source: audited v0.8 model summary", ha="right", fontsize=9.2, color=GRAY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.027, pad=0.03)
    cbar.set_label("Metric value (0–1)", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.23, right=0.92, top=0.84, bottom=0.25)
    save(fig, "v08-cross-model-comparison", tight=False)


def make_v01_v07_figure() -> None:
    with (OUT / "qwen25-v01-v07-validation.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    versions = [r["version"] for r in rows]
    recipes = [r["recipe"] for r in rows]
    x = np.arange(len(rows))
    validity = np.array([int(r["valid"]) / int(r["examples"]) for r in rows]) * 100
    correctness = np.array([float(r["correctness_accuracy"]) for r in rows]) * 100
    correctness_f1 = np.array([float(r["correctness_macro_f1"]) for r in rows]) * 100
    taxonomy_f1 = np.array([float(r["taxonomy_macro_f1"]) for r in rows]) * 100
    first_error = np.array([float(r["first_error_exact"]) for r in rows]) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.8, 9.0), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18})
    series_top = [
        ("Structured validity", validity, NAVY, "o"),
        ("Correctness accuracy", correctness, TEAL, "s"),
        ("Correctness macro-F1", correctness_f1, ORANGE, "D"),
    ]
    series_bottom = [
        ("Taxonomy macro-F1", taxonomy_f1, RED, "o"),
        ("First-error exact", first_error, BLUE, "s"),
    ]

    for ax, series in ((ax1, series_top), (ax2, series_bottom)):
        for label, values, color, marker in series:
            ax.plot(x, values, label=label, color=color, marker=marker, markersize=7, linewidth=2.6)
            for xi, value in zip(x, values):
                ax.annotate(f"{value:.1f}", (xi, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=color)
        ax.set_ylim(0, 106)
        ax.set_ylabel("Validation score (%)", fontsize=11, fontweight="bold")
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color="#D7E0E6", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="lower left", ncol=len(series), frameon=False, fontsize=10)

    best_correct = int(np.argmax(correctness))
    best_local = int(np.argmax(first_error))
    ax1.annotate("Best correctness", xy=(best_correct, correctness[best_correct]), xytext=(best_correct + 0.5, correctness[best_correct] - 18), arrowprops=dict(arrowstyle="->", color=GRAY), fontsize=9.5, color=GRAY, fontweight="bold")
    ax2.annotate("Best localization", xy=(best_local, first_error[best_local]), xytext=(best_local + 0.6, first_error[best_local] + 10), arrowprops=dict(arrowstyle="->", color=GRAY), fontsize=9.5, color=GRAY, fontweight="bold")

    tick_labels = [f"{v}\n{textwrap.fill(r, width=18)}" for v, r in zip(versions, recipes)]
    ax2.set_xticks(x, tick_labels, fontsize=9.0)
    ax2.tick_params(axis="x", pad=10, length=0)
    ax2.set_xlabel("Experiment version and main training change", fontsize=11, fontweight="bold", labelpad=22)

    # The y-axis labels make the chart body asymmetric within the canvas.
    fig.suptitle("Fine-Tuning Progression", x=0.525, y=0.965, ha="center", fontsize=24, fontweight="bold", color=NAVY)
    footer_lines = [
        "• Validity = the model returned usable JSON.",
        "• Correctness = the model assigned the right grade.",
        "• Taxonomy = the model identified the right error type.",
        "• First-error = the model found the first wrong step.",
        "• Every version used the same 48 validation problems.",
        "• Prompt-only zero-shot and few-shot tests came before v0.1; v0.1 is the first QLoRA fine-tune.",
    ]
    for line_y, line in zip((0.175, 0.148, 0.121, 0.094, 0.067, 0.040), footer_lines):
        fig.text(0.09, line_y, line, fontsize=9.3, color=GRAY)
    fig.text(0.96, 0.018, "Sources: v0.1–v0.7 validation reports", ha="right", fontsize=9.2, color=GRAY)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.88, bottom=0.31)
    save(fig, "qwen25-v01-v07-validation-progression", tight=False)


if __name__ == "__main__":
    make_methods_figure()
    make_cross_model_figure()
    make_v01_v07_figure()
    print(f"Wrote poster assets to {OUT}")
