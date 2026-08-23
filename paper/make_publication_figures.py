"""Regenerate paper figures from frozen experiment CSV files.

This script changes presentation only. It reads the archived final_v3 tables and
writes publication copies into ``paper/figures``; it never edits experiment
outputs or recomputes model results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "final_v3" / "results"
SEMANTIC = ROOT / "outputs" / "final_v3" / "semantic"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

PUBLIC_NAMES = {
    "relative_raw": "Raw steering",
    "raw": "Raw steering",
    "sae_calibrated": "SAE reconstruction",
    "conditioned_reconstruction": "Conditioned reconstruction",
    "conditioned_kl": "Conditioned KL",
    "conditioned_kl_retention": "KL + retention",
    "hard_projected_conditioned_kl_retention": "Hard projection",
}

METHOD_ORDER = [
    "relative_raw",
    "sae_calibrated",
    "conditioned_reconstruction",
    "conditioned_kl",
    "conditioned_kl_retention",
]

STYLES = {
    "relative_raw": ("o", "-"),
    "raw": ("o", "-"),
    "sae_calibrated": ("s", "--"),
    "conditioned_reconstruction": ("^", "-."),
    "conditioned_kl": ("D", ":"),
    "conditioned_kl_retention": ("P", "-"),
    "hard_projected_conditioned_kl_retention": ("X", "--"),
}


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout(pad=0.45)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _line(ax: plt.Axes, frame: pd.DataFrame, method: str, y: str) -> None:
    rows = frame[frame["method"] == method].sort_values("relative_strength")
    marker, linestyle = STYLES[method]
    ax.plot(
        rows["relative_strength"],
        rows[y],
        marker=marker,
        linestyle=linestyle,
        linewidth=1.5,
        markersize=4.2,
        label=PUBLIC_NAMES[method],
    )


def strength_figures() -> None:
    frame = pd.read_csv(RESULTS / "ablation_summary.csv")
    frame = frame[frame["evaluation_split"] == "validation"]
    specs = [
        ("mean_delta_nll", r"$\Delta$NLL $\downarrow$", "strength_delta_nll"),
        (
            "mean_concept_retention",
            r"Target-feature retention $\uparrow$",
            "strength_retention",
        ),
    ]
    for y, ylabel, stem in specs:
        fig, ax = plt.subplots(figsize=(4.9, 3.25))
        for method in METHOD_ORDER:
            _line(ax, frame, method, y)
        ax.axhline(0.0 if y == "mean_delta_nll" else 1.0, color="0.65", lw=0.8)
        ax.set_xlabel("Relative steering strength")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(frameon=False, ncol=2)
        _save(fig, stem)


def sentiment_figure() -> None:
    frame = pd.read_csv(RESULTS / "cross_concept_aggregate.csv")
    methods = [
        "relative_raw",
        "sae_calibrated",
        "conditioned_reconstruction",
        "conditioned_kl",
        "conditioned_kl_retention",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.05))
    for method in methods:
        _line(axes[0], frame, method, "mean_delta_nll")
        _line(axes[1], frame, method, "mean_concept_retention")
    axes[0].set_ylabel(r"$\Delta$NLL $\downarrow$")
    axes[1].set_ylabel(r"Target-feature retention $\uparrow$")
    axes[1].axhline(1.0, color="0.65", lw=0.8)
    for ax in axes:
        ax.set_xlabel("Relative steering strength")
        ax.grid(alpha=0.2, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.subplots_adjust(top=0.80)
    _save(fig, "cross_concept_sentiment_confirmation")


def hard_projection_figure() -> None:
    holdout = pd.read_csv(RESULTS / "token_holdout_or_replication.csv")
    holdout = (
        holdout.groupby(["method", "relative_strength"], as_index=False)
        .agg(mean_delta_nll=("delta_nll", "mean"), mean_concept_retention=("concept_retention", "mean"))
    )
    sentiment = pd.read_csv(RESULTS / "cross_concept_aggregate.csv")
    methods = ["conditioned_kl_retention", "hard_projected_conditioned_kl_retention"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for method in methods:
        _line(axes[0], holdout, method, "mean_delta_nll")
        _line(axes[1], sentiment, method, "mean_concept_retention")
    axes[0].set_title("Unseen SAE directions")
    axes[0].set_ylabel(r"$\Delta$NLL $\downarrow$")
    axes[1].set_title("Independent sentiment direction")
    axes[1].set_ylabel(r"Target-feature retention $\uparrow$")
    axes[1].axhline(1.0, color="0.65", lw=0.8)
    for ax in axes:
        ax.set_xlabel("Relative steering strength")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(frameon=False)
    _save(fig, "hard_projection_vs_soft_retention")


def semantic_scatter() -> None:
    frame = pd.read_csv(SEMANTIC / "semantic_scores.csv")
    frame = frame[frame["experiment_role"] == "primary"].copy()
    methods = ["raw", "sae_calibrated", "conditioned_reconstruction", "conditioned_kl", "conditioned_kl_retention"]
    fig, ax = plt.subplots(figsize=(6.25, 4.0))
    for method in methods:
        rows = frame[frame["method"] == method]
        marker, _ = STYLES[method]
        ax.scatter(
            rows["target_sae_activation"],
            rows["semantic_concept_score"],
            marker=marker,
            s=24,
            alpha=0.52,
            linewidths=0.3,
            label=PUBLIC_NAMES[method],
        )
    ax.set_xlabel("Target SAE activation")
    ax.set_ylabel("External semantic score")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(frameon=False, ncol=2)
    _save(fig, "sae_vs_external_semantic_score")


def appendix_figures() -> None:
    harmfulness = pd.read_csv(RESULTS / "harmfulness.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.hist(harmfulness["mean_kl"], bins=20, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"Mean KL(clean $\Vert$ modified) $\downarrow$")
    ax.set_ylabel("Training directions")
    _save(fig, "harmfulness_distribution")

    neighbors = pd.read_csv(RESULTS / "natural_neighbor_diagnostics.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for method in ["relative_raw", "sae_calibrated", "conditioned_full"]:
        rows = neighbors[neighbors["method"] == method]
        name = {"relative_raw": "Raw steering", "sae_calibrated": "SAE reconstruction", "conditioned_full": "Conditioned model"}[method]
        ax.scatter(rows["knn_distance"], rows["delta_nll"], s=10, alpha=0.35, label=name)
    ax.set_xlabel(r"kNN distance $\downarrow$")
    ax.set_ylabel(r"$\Delta$NLL $\downarrow$")
    ax.legend(frameon=False)
    _save(fig, "knn_distance_vs_delta_nll")

    aggregate = pd.read_csv(SEMANTIC / "semantic_aggregate.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for method in ["raw", "sae_calibrated", "conditioned_reconstruction", "conditioned_kl", "conditioned_kl_retention"]:
        rows = aggregate[aggregate["method"] == method].sort_values("strength")
        marker, linestyle = STYLES[method]
        ax.plot(rows["strength"], rows["mean_semantic_concept_score"], marker=marker, linestyle=linestyle, lw=1.4, ms=4, label=PUBLIC_NAMES[method])
    ax.set_xlabel("Relative steering strength")
    ax.set_ylabel(r"External semantic score $\uparrow$")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, ncol=2)
    _save(fig, "semantic_score_vs_strength")

    negative = pd.read_csv(SEMANTIC / "negative_control.csv")
    scores = pd.read_csv(SEMANTIC / "semantic_scores.csv")
    primary = scores[(scores["experiment_role"] == "primary") & (scores["method"] == "raw")].groupby("strength", as_index=False)["semantic_concept_score"].mean()
    neg_y = "semantic_concept_score" if "semantic_concept_score" in negative.columns else "negative_control_score"
    negative = negative.groupby("strength", as_index=False)[neg_y].mean()
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(primary["strength"], primary["semantic_concept_score"], marker="o", lw=1.5, label="Greeting direction")
    ax.plot(negative["strength"], negative[neg_y], marker="s", linestyle="--", lw=1.5, label="Control direction")
    ax.set_xlabel("Relative steering strength")
    ax.set_ylabel(r"External semantic score $\uparrow$")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False)
    _save(fig, "greeting_vs_control_direction")


def main() -> None:
    _configure()
    strength_figures()
    sentiment_figure()
    hard_projection_figure()
    semantic_scatter()
    appendix_figures()
    print(f"Wrote publication figures to {FIGURES}")


if __name__ == "__main__":
    main()
