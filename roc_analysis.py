#!/usr/bin/env python3
"""Generate ROC curves and Recall@FPR analysis for NARCBench.

Produces:
  - Per-tier ROC curve PDFs/PNGs (one per NARCBench tier)
  - Combined multi-panel ROC figure
  - Recall@1%FPR and Recall@5%FPR summary table (printed to stdout)

Methodology follows Goldowsky-Dill et al. (2025):
  - Raw projected scores (not probabilities) for threshold-free methods
  - 1000-threshold sweep for smooth ROC curves
  - Recall@FPR via quantile threshold on control scores

Usage:
  python roc_analysis.py                     # defaults: layer 28, all tiers
  python roc_analysis.py --layer 30          # single layer
  python roc_analysis.py --tiers transfer    # single tier
  python roc_analysis.py --out-dir results/   # custom output directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from reproduce.reproduce_LR import (
    LAYERS,
    METHOD_KEYS,
    build_groups,
    compute_scores,
    load_data,
    train_deception_probe,
    _run_labels,
    _train_asymmetry_classifier,
    _train_concat_classifier,
)

# ── Constants ──────────────────────────────────────────────────────────────

DATA_DIR = config.ACTIVATIONS_ROOT / "qwen3_32b"

TIER_DIRS = {
    "core": DATA_DIR / "core",
    "transfer": DATA_DIR / "transfer",
    "stego": DATA_DIR / "stego",
}

TIER_PHASES = {
    "core": {"phase": "discussion", "avg_rounds": True},
    "transfer": {"phase": "final", "avg_rounds": False},
    "stego": {"phase": "final", "avg_rounds": False},
}

METHOD_LABELS = {
    "sorted_concat": "Sorted Concatenation",
    "activation_alignment": "Activation Alignment",
    "peak_suspicion": "Peak Suspicion",
    "deception_split": "Deception Split",
    "asymmetry_probe": "Asymmetry Probe",
}

COLORS = {
    "sorted_concat": "#C0392B",
    "activation_alignment": "#E74C3C",
    "peak_suspicion": "#E67E22",
    "deception_split": "#F39C8B",
    "asymmetry_probe": "#D4577A",
}


# ── ROC computation ────────────────────────────────────────────────────────

def compute_roc(control_scores: np.ndarray, collusion_scores: np.ndarray,
                n_thresholds: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Compute FPR/TPR curve by sweeping thresholds over raw scores."""
    all_scores = np.concatenate([control_scores, collusion_scores])
    thresholds = np.linspace(all_scores.min() - 0.1,
                             all_scores.max() + 0.1,
                             n_thresholds)
    fprs = np.array([(control_scores > t).mean() for t in thresholds])
    tprs = np.array([(collusion_scores > t).mean() for t in thresholds])
    return fprs, tprs


def recall_at_fpr(control_scores: np.ndarray, collusion_scores: np.ndarray,
                  target_fpr: float = 0.01) -> float:
    """Compute recall (TPR) at a target FPR via quantile threshold."""
    threshold = np.quantile(control_scores, 1 - target_fpr)
    return float((collusion_scores > threshold).mean())


# ── Score collection ───────────────────────────────────────────────────────

def collect_tier_scores(
    groups: dict, rids: list, scaler, w,
    core_scores: dict | None = None,
    core_y: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect per-method scores split into collusion and control.

    For threshold-free methods (peak_suspicion, deception_split,
    activation_alignment), scores are raw projections. For classifier-based
    methods (asymmetry_probe, sorted_concat), scores are class probabilities
    from classifiers trained on core_scores.

    Returns:
        {method_key: {"collusion": array, "control": array}}
    """
    scores = compute_scores(groups, rids, scaler, w)
    y = _run_labels(groups, rids)

    collusion_mask = y == 1
    control_mask = y == 0

    result = {}

    # Threshold-free methods: use raw scores directly
    for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
        result[key] = {
            "collusion": scores[key][collusion_mask],
            "control": scores[key][control_mask],
        }

    # Classifier-based methods: train on Core, predict probabilities
    if core_scores is not None and core_y is not None:
        asym_scaler, asym_clf = _train_asymmetry_classifier(core_scores, core_y)
        asym_proba = asym_clf.predict_proba(
            asym_scaler.transform(scores["asymmetry_probe"]))[:, 1]
        result["asymmetry_probe"] = {
            "collusion": asym_proba[collusion_mask],
            "control": asym_proba[control_mask],
        }

        concat_pca, concat_scaler, concat_clf = _train_concat_classifier(
            core_scores, core_y)
        concat_proba = concat_clf.predict_proba(
            concat_scaler.transform(
                concat_pca.transform(scores["sorted_concat"])))[:, 1]
        result["sorted_concat"] = {
            "collusion": concat_proba[collusion_mask],
            "control": concat_proba[control_mask],
        }

    return result


# ── Plotting ───────────────────────────────────────────────────────────────

def _apply_style() -> None:
    """Apply clean plot style (applied once at startup)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
    })


def plot_roc_single(tier_scores: dict, tier_name: str, out_dir: Path) -> None:
    """Save a single-panel ROC plot for one tier (PDF + PNG)."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for method in METHOD_KEYS:
        if method not in tier_scores:
            continue
        fprs, tprs = compute_roc(
            tier_scores[method]["control"],
            tier_scores[method]["collusion"])
        auroc = auc(fprs, tprs)
        ax.plot(fprs * 100, tprs * 100,
                color=COLORS[method], linewidth=1.5,
                label=f"{METHOD_LABELS[method]} ({auroc:.2f})")

    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.3)
    ax.set(xlim=(0, 100), ylim=(0, 102),
           xlabel="False Positive Rate (%)",
           ylabel="True Positive Rate (%)",
           title=f"NARCBench-{tier_name}")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    plt.tight_layout()

    stem = f"roc_{tier_name.lower()}"
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.{{pdf,png}}")


def plot_roc_combined(all_tier_scores: dict, out_dir: Path,
                      tiers: list[str] | None = None) -> None:
    """Save a multi-panel ROC plot with one panel per tier + shared legend."""
    if tiers is None:
        tiers = [t for t in ["transfer", "stego"] if t in all_tier_scores]
    n = len(tiers)
    if n == 0:
        return

    fig = plt.figure(figsize=(3.5 * n + 1.5, 2.3))
    gs = fig.add_gridspec(1, n + 1,
                          width_ratios=[1] * n + [0.45],
                          wspace=0.35)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    for i in range(1, n):
        axes[i].sharey(axes[0])

    for ax, tier_key in zip(axes, tiers):
        tier_scores = all_tier_scores[tier_key]
        for method in METHOD_KEYS:
            if method not in tier_scores:
                continue
            fprs, tprs = compute_roc(
                tier_scores[method]["control"],
                tier_scores[method]["collusion"])
            auroc = auc(fprs, tprs)
            ax.plot(fprs * 100, tprs * 100,
                    color=COLORS[method], linewidth=1.5,
                    label=f"{METHOD_LABELS[method]} ({auroc:.2f})")

        ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.3)
        ax.set(xlim=(0, 100), ylim=(0, 102),
               xlabel="False Positive Rate (%)",
               title=f"NARCBench-{tier_key.capitalize()}")
        ax.set_aspect("equal")

    axes[0].set_ylabel("True Positive Rate (%)")
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    leg_ax = fig.add_subplot(gs[0, n])
    leg_ax.axis("off")
    leg_ax.legend(handles, labels, loc="center left",
                  fontsize=7, frameon=False, borderpad=0)

    for ext in ("pdf", "png"):
        path = out_dir / f"roc_combined.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved roc_combined.{{pdf,png}}")


# ── Summary table ──────────────────────────────────────────────────────────

def print_summary(all_tier_scores: dict, tiers: list[str]) -> None:
    """Print AUROC and Recall@FPR table to stdout."""
    header = f"{'Method':>25}  {'AUROC':>8}  {'R@1%FPR':>8}  {'R@5%FPR':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for tier_key in tiers:
        if tier_key not in all_tier_scores:
            continue
        tier_scores = all_tier_scores[tier_key]
        print(f"\n  {tier_key.capitalize()}")

        for method in METHOD_KEYS:
            if method not in tier_scores:
                continue
            ctrl = tier_scores[method]["control"]
            col = tier_scores[method]["collusion"]
            fprs, tprs = compute_roc(ctrl, col)
            auroc_val = auc(fprs, tprs)
            r1 = recall_at_fpr(ctrl, col, 0.01)
            r5 = recall_at_fpr(ctrl, col, 0.05)
            print(f"    {METHOD_LABELS[method]:>23}  {auroc_val:>8.3f}  {r1:>8.3f}  {r5:>8.3f}")

    print("=" * len(header))


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ROC curves and Recall@FPR analysis for NARCBench.")
    p.add_argument("--layer", type=int, default=28,
                   help="Layer to use for deception probe (default: 28)")
    p.add_argument("--tiers", nargs="+", default=["transfer", "stego"],
                   choices=["core", "transfer", "stego"],
                   help="Which tiers to evaluate (default: all)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: results/figures)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else config.RESULTS_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data for requested tiers
    print("Loading data...")
    data = {}
    for tier in args.tiers:
        data[tier] = load_data(TIER_DIRS[tier])

    # Train deception probe on Core (always needed)
    if "core" not in data:
        data["core"] = load_data(TIER_DIRS["core"])
    core_meta, core_npz = data["core"]

    print(f"Training deception probe at layer {args.layer}...")
    scaler, w = train_deception_probe(core_meta, core_npz, args.layer)

    # Build groups and collect scores per tier
    print("Computing scores...")
    core_groups = build_groups(core_meta, core_npz, args.layer,
                               **TIER_PHASES["core"])
    core_rids = sorted(core_groups.keys())
    core_scores = compute_scores(core_groups, core_rids, scaler, w)
    core_y = _run_labels(core_groups, core_rids)

    all_tier_scores = {}
    for tier in args.tiers:
        meta, npz = data[tier]
        groups = build_groups(meta, npz, args.layer, **TIER_PHASES[tier])
        rids = sorted(groups.keys())
        all_tier_scores[tier] = collect_tier_scores(
            groups, rids, scaler, w, core_scores, core_y)

    # Print summary table
    print_summary(all_tier_scores, args.tiers)

    # Generate plots
    _apply_style()
    print("\nGenerating ROC curves...")
    for tier in args.tiers:
        plot_roc_single(all_tier_scores[tier], tier.capitalize(), out_dir)
    plot_roc_combined(all_tier_scores, out_dir, args.tiers)

    print("\nDone.")


if __name__ == "__main__":
    main()
