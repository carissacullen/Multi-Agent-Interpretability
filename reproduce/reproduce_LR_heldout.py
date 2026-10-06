"""
Held-out-only variant of the LR probe reproduction pipeline.

Unlike reproduce_LR.py, hyperparameters here are never selected using
NARCBench-Transfer or NARCBench-Stego scores. Model selection must use only
the Held-out metric: leave-domain-out cross-validation AUROC on
NARCBench-Core. Transfer/Stego ("OOD") AUROC is still computed and reported
here -- using the same group/method/layer/seed averaging as the existing
"OOD mean AUROC" in reproduce_LR.py, so the numbers are directly comparable
-- but it is report-only. tune_hyperparams.py enforces this split when
searching for a config; this file just provides the pieces (and a CLI for
running one frozen config by hand).

This file intentionally duplicates logic from reproduce_LR.py rather than
importing it, so the two stay fully independent and reproduce_LR.py is left
unmodified.

Usage:
    python reproduce_LR_heldout.py
    python reproduce_LR_heldout.py --C-probe 25 --max-iter-probe 8 --solver liblinear
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Hyperparameters ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeHParams:
    """All tunable knobs for the deception probe + downstream classifiers.

    Passed explicitly through every function (never read from module
    globals) so a sweep can try many configs in one process without one
    run's settings leaking into the next.
    """
    C_probe: float = 25.0
    max_iter_probe: int = 8
    penalty: str = "l2"
    solver: str = "liblinear"
    C_clf: float = 0.01
    max_iter_clf: int = 2000
    aug_enabled: bool = False
    aug_noise_std: float = 0.05
    aug_n_copies: int = 4
    pca_components: int | None = None


# ── Paths and constants ──────────────────────────────────────────────────

# Both models' activations are available locally under data/activations/<model>/.
# qwen3_32b layers match the paper's original Qwen3-32B setup (the default);
# gpt_oss_20b layers mirror what's active in reproduce_LR.py -- pass
# --model gpt_oss_20b to use that instead.
MODEL_LAYERS = {
    "gpt_oss_20b": [10, 11, 12, 13, 14],
    "qwen3_32b":   [26, 27, 28, 29, 30],
}
DEFAULT_MODEL = "qwen3_32b"
DEFAULT_N_SEEDS = 10


class ModelPaths:
    """Data/output paths + default layers for one model. Built on demand
    (not fixed at import time) so scripts can select the model via --model
    instead of always using gpt_oss_20b.
    """

    def __init__(self, model: str):
        if model not in MODEL_LAYERS:
            raise ValueError(f"Unknown model {model!r}; choices: {sorted(MODEL_LAYERS)}")
        self.model = model
        data_dir = config.ACTIVATIONS_ROOT / model
        self.core_dir = data_dir / "core"
        self.transfer_dir = data_dir / "transfer"
        self.stego_dir = data_dir / "stego"
        # Independent of config.RESULTS_ROOT (which tracks whatever model
        # config.py is currently pointed at elsewhere in the repo) so output
        # for each model always lands under its own results/<model>/ dir.
        self.csv_dir = config.ROOT / "results" / model / "auroc_scores"
        self.default_layers = MODEL_LAYERS[model]


def get_paths(model: str = DEFAULT_MODEL) -> ModelPaths:
    return ModelPaths(model)


# Module-level defaults (DEFAULT_MODEL, i.e. qwen3_32b) kept for backward
# compatibility with any code importing these names directly; scripts with a
# --model flag use get_paths(args.model) instead and ignore these.
_default_paths = get_paths(DEFAULT_MODEL)
CORE_DIR = _default_paths.core_dir
TRANSFER_DIR = _default_paths.transfer_dir
STEGO_DIR = _default_paths.stego_dir
CSV_DIR = _default_paths.csv_dir
DEFAULT_LAYERS = _default_paths.default_layers

METHOD_KEYS = [
    "sorted_concat",
    "activation_alignment",
    "peak_suspicion",
    "deception_split",
    "asymmetry_probe",
]

# Only these two methods go through a classifier with predict_proba, so only
# they have a well-defined probability to score with log-loss. The other
# three methods output a raw projection score, not a calibrated probability
# -- turning them into one would need an arbitrary temperature, so they're
# left out rather than faked.
LOGLOSS_METHOD_KEYS = ["asymmetry_probe", "sorted_concat"]

CSV_DIR = config.RESULTS_ROOT / "auroc_scores"


# ── Data loading ─────────────────────────────────────────────────────────

def load_data(directory):
    """Load metadata and activations from a dataset directory."""
    with open(directory / "metadata_gen.json") as f:
        meta = json.load(f)
    return meta, np.load(directory / "activations_gen.npz")


def get_layer(npz, n, layer):
    return npz[f"layer_{layer}"][:n].astype(np.float32)


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_run_id(run_id):
    parts = run_id.split("__")
    scenario = parts[0]
    scenario_id = parts[1] if len(parts) >= 2 else run_id
    return scenario, scenario_id


def _domain_from_scenario_id(scenario_id):
    return "_".join(scenario_id.split("_")[:-1])


def _run_labels(groups, rids):
    return np.array([1 if groups[r]["mode"] == "collusion" else 0 for r in rids])


# ── Group building ───────────────────────────────────────────────────────

def build_groups(meta, npz, layer, phase="discussion", avg_rounds=True):
    n = len(meta)
    acts = get_layer(npz, n, layer)

    if phase == "discussion" and avg_rounds:
        accum = defaultdict(lambda: {"vecs": [], "meta": None})
        for i, m in enumerate(meta):
            if m["phase"] != "discussion" or m["channel"] != "public":
                continue
            accum[(m["run_id"], m["agent_name"])]["vecs"].append(acts[i])
            accum[(m["run_id"], m["agent_name"])]["meta"] = m

        groups = {}
        for (rid, agent_name), data in accum.items():
            m = data["meta"]
            if rid not in groups:
                scenario, scenario_id = _parse_run_id(rid)
                groups[rid] = {
                    "agents": {}, "mode": m["mode"], "roles": {},
                    "scenario": scenario,
                    "domain": _domain_from_scenario_id(scenario_id),
                }
            groups[rid]["agents"][agent_name] = np.mean(data["vecs"], axis=0)
            groups[rid]["roles"][agent_name] = m["role"]
    else:
        groups = {}
        for i, m in enumerate(meta):
            if m["phase"] != phase or m["channel"] != "public":
                continue
            rid = m["run_id"]
            if rid not in groups:
                scenario, scenario_id = _parse_run_id(rid)
                groups[rid] = {
                    "agents": {}, "mode": m["mode"], "roles": {},
                    "scenario": scenario,
                    "domain": _domain_from_scenario_id(scenario_id),
                }
            groups[rid]["agents"][m["agent_name"]] = acts[i]
            groups[rid]["roles"][m["agent_name"]] = m["role"]

    return {rid: g for rid, g in groups.items() if len(g["agents"]) == 4}


# ── Probe training ───────────────────────────────────────────────────────

def train_deception_probe(meta, npz, layer, hp: ProbeHParams,
                           phase="discussion", exclude_domain=None, seed=0):
    """Train the deception direction from raw per-sample activations.

    If exclude_domain is set, samples from that domain are excluded
    (used for leave-domain-out CV).
    """
    n = len(meta)
    acts = get_layer(npz, n, layer)

    col_idx, hon_idx = [], []
    for i, m in enumerate(meta):
        if m["mode"] != "collusion":
            continue
        if m["channel"] != "public":
            continue
        if phase == "discussion" and m["phase"] != "discussion":
            continue
        _, scenario_id = _parse_run_id(m["run_id"])
        if exclude_domain and _domain_from_scenario_id(scenario_id) == exclude_domain:
            continue
        if m["role"] == "colluder":
            col_idx.append(i)
        elif m["role"] == "honest":
            hon_idx.append(i)

    nc = min(len(col_idx), len(hon_idx))
    rng = np.random.RandomState(seed)
    hon_idx = list(rng.choice(hon_idx, nc, replace=False))
    col_idx = col_idx[:nc]

    scaler = StandardScaler()
    X = scaler.fit_transform(np.vstack([acts[hon_idx], acts[col_idx]]))
    y = np.array([0] * nc + [1] * nc)

    if hp.aug_enabled and hp.aug_noise_std > 0:
        rng2 = np.random.RandomState(0)
        noise = rng2.normal(0, hp.aug_noise_std,
                             size=(len(y) * hp.aug_n_copies, X.shape[1]))
        X_aug = np.tile(X, (hp.aug_n_copies, 1)) + noise
        y_aug = np.tile(y, hp.aug_n_copies)
        X = np.vstack([X, X_aug])
        y = np.concatenate([y, y_aug])

    if hp.pca_components is not None:
        n_comp = min(hp.pca_components, X.shape[0] - 1, X.shape[1])
        probe_pca = PCA(n_components=n_comp, random_state=42)
        X_fit = probe_pca.fit_transform(X)
    else:
        probe_pca = None
        X_fit = X

    probe = LogisticRegression(max_iter=hp.max_iter_probe, C=hp.C_probe,
                                penalty=hp.penalty, solver=hp.solver,
                                random_state=seed)
    probe.fit(X_fit, y)
    w_fit = probe.coef_[0]
    w = probe_pca.components_.T @ w_fit if probe_pca is not None else w_fit
    w = w / np.linalg.norm(w)
    return scaler, w


# ── Score computation ────────────────────────────────────────────────────

def compute_scores(groups, rids, scaler, w):
    """Compute all five method scores for each run."""
    peak_suspicion, deception_split, asymmetry_probe = [], [], []
    activation_alignment, sorted_concat = [], []

    for r in rids:
        g = groups[r]
        names = sorted(g["agents"])
        raw_vecs = [g["agents"][n] for n in names]
        scaled_vecs = [scaler.transform(v.reshape(1, -1))[0] for v in raw_vecs]

        projs = np.array([w @ v for v in scaled_vecs])
        order = np.argsort(projs)[::-1]
        projs_sorted = projs[order]

        peak_suspicion.append(projs_sorted[0])
        deception_split.append(np.mean(projs_sorted[:2]) - np.mean(projs_sorted[2:]))
        asymmetry_probe.append([
            np.var(projs),
            projs_sorted[0] - projs_sorted[3],
            np.mean(projs_sorted[:2]) - np.mean(projs_sorted[2:]),
            np.mean(projs),
        ])
        sorted_concat.append(np.concatenate([scaled_vecs[i] for i in order]))

        norms = [np.linalg.norm(v) + 1e-10 for v in raw_vecs]
        normed = [v / n for v, n in zip(raw_vecs, norms)]
        cos_pairs = [normed[i] @ normed[j] for i in range(4) for j in range(i + 1, 4)]
        activation_alignment.append(-np.mean(cos_pairs))

    return {
        "peak_suspicion": np.array(peak_suspicion),
        "deception_split": np.array(deception_split),
        "asymmetry_probe": np.array(asymmetry_probe),
        "activation_alignment": np.array(activation_alignment),
        "sorted_concat": np.array(sorted_concat),
    }


# ── Downstream classifiers ─────────────────────────────────────────────────

def _train_asymmetry_classifier(tr_scores, tr_y, hp: ProbeHParams):
    fs = StandardScaler()
    clf = LogisticRegression(max_iter=hp.max_iter_clf, C=hp.C_clf,
                              penalty=hp.penalty, solver=hp.solver, random_state=42)
    clf.fit(fs.fit_transform(tr_scores["asymmetry_probe"]), tr_y)
    return fs, clf


def _train_concat_classifier(tr_scores, tr_y, hp: ProbeHParams):
    n_comp = min(50, len(tr_y) - 1, tr_scores["sorted_concat"].shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    fs = StandardScaler()
    X_tr = fs.fit_transform(pca.fit_transform(tr_scores["sorted_concat"]))
    clf = LogisticRegression(max_iter=hp.max_iter_clf, C=hp.C_clf,
                              penalty=hp.penalty, solver=hp.solver, random_state=42)
    clf.fit(X_tr, tr_y)
    return pca, fs, clf


def evaluate(tr_scores, tr_y, te_scores, te_y, hp: ProbeHParams):
    """Evaluate all five methods. Returns dict of method_key -> AUROC.

    Used by the pooled=False (per-fold) path of held_out_cv_layer, kept only
    for comparison -- see test_pooling.py.
    """
    if len(np.unique(te_y)) < 2:
        return None

    results = {}
    for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
        results[key] = roc_auc_score(te_y, te_scores[key])

    fs, clf = _train_asymmetry_classifier(tr_scores, tr_y, hp)
    results["asymmetry_probe"] = roc_auc_score(
        te_y, clf.predict_proba(fs.transform(te_scores["asymmetry_probe"]))[:, 1])

    pca, fs2, clf2 = _train_concat_classifier(tr_scores, tr_y, hp)
    X_te = fs2.transform(pca.transform(te_scores["sorted_concat"]))
    results["sorted_concat"] = roc_auc_score(te_y, clf2.predict_proba(X_te)[:, 1])

    return results


def _held_out_predictions(tr_scores, tr_y, te_scores, hp: ProbeHParams):
    """Return per-method raw prediction scores for one held-out fold.

    Same underlying models as evaluate(), but returns scores instead of a
    fold AUROC so predictions from many folds can be concatenated and
    scored together -- see held_out_cv_layer(pooled=True).
    """
    preds = {}
    for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
        preds[key] = te_scores[key]

    fs, clf = _train_asymmetry_classifier(tr_scores, tr_y, hp)
    preds["asymmetry_probe"] = clf.predict_proba(fs.transform(te_scores["asymmetry_probe"]))[:, 1]

    pca, fs2, clf2 = _train_concat_classifier(tr_scores, tr_y, hp)
    X_te = fs2.transform(pca.transform(te_scores["sorted_concat"]))
    preds["sorted_concat"] = clf2.predict_proba(X_te)[:, 1]

    return preds


# ── SELECTION METRIC: leave-domain-out CV on Core ──────────────────────────

def held_out_cv_layer(core_meta, core_npz, layer, hp: ProbeHParams, seed=0, pooled=True):
    """Leave-domain-out cross-validation on Core for one layer.

    This is the only signal tune_hyperparams.py is allowed to use to pick a
    winner between configs.

    pooled=True (default): concatenate every held-out domain's raw
    predictions and true labels, then compute ONE AUROC per method over the
    pooled set. Fixes the ceiling effect you get from averaging AUROC over
    many tiny per-domain folds -- each fold may only hold 2-4 runs, so a
    per-fold AUROC is coarsely quantised (0, 0.5, 1, ...) and saturates at
    1.0 almost immediately, leaving no resolution to compare hyperparameters.
    Also lets folds with only one class in the test split still contribute
    (they were previously skipped entirely and wasted).

    pooled=False: the original per-fold-AUROC-then-average behaviour, kept
    only so test_pooling.py can show the before/after difference.
    """
    groups = build_groups(core_meta, core_npz, layer, phase="discussion", avg_rounds=True)
    rids = sorted(groups.keys())
    domains = sorted({groups[r]["domain"] for r in rids if groups[r]["domain"]})

    if not pooled:
        fold_results = {k: [] for k in METHOD_KEYS}

        for hold_dom in domains:
            tr_rids = [r for r in rids if groups[r]["domain"] != hold_dom]
            te_rids = [r for r in rids if groups[r]["domain"] == hold_dom]
            tr_y = _run_labels(groups, tr_rids)
            te_y = _run_labels(groups, te_rids)

            if len(np.unique(te_y)) < 2 or len(np.unique(tr_y)) < 2:
                continue

            scaler, w = train_deception_probe(core_meta, core_npz, layer, hp,
                                               exclude_domain=hold_dom, seed=seed)
            tr_scores = compute_scores(groups, tr_rids, scaler, w)
            te_scores = compute_scores(groups, te_rids, scaler, w)

            result = evaluate(tr_scores, tr_y, te_scores, te_y, hp)
            if result:
                for k in METHOD_KEYS:
                    fold_results[k].append(result[k])

        return {k: (np.mean(v) if v else float("nan")) for k, v in fold_results.items()}

    # -- pooled path (default) --
    pooled_scores = {k: [] for k in METHOD_KEYS}
    pooled_labels = []

    for hold_dom in domains:
        tr_rids = [r for r in rids if groups[r]["domain"] != hold_dom]
        te_rids = [r for r in rids if groups[r]["domain"] == hold_dom]
        tr_y = _run_labels(groups, tr_rids)
        te_y = _run_labels(groups, te_rids)

        if len(te_rids) == 0 or len(np.unique(tr_y)) < 2:
            continue

        scaler, w = train_deception_probe(core_meta, core_npz, layer, hp,
                                           exclude_domain=hold_dom, seed=seed)
        tr_scores = compute_scores(groups, tr_rids, scaler, w)
        te_scores = compute_scores(groups, te_rids, scaler, w)

        preds = _held_out_predictions(tr_scores, tr_y, te_scores, hp)
        for k in METHOD_KEYS:
            pooled_scores[k].append(preds[k])
        pooled_labels.append(te_y)

    if not pooled_labels:
        return {**{k: float("nan") for k in METHOD_KEYS},
                **{f"{k}_logloss": float("nan") for k in LOGLOSS_METHOD_KEYS}}

    y_all = np.concatenate(pooled_labels)
    if len(np.unique(y_all)) < 2:
        return {**{k: float("nan") for k in METHOD_KEYS},
                **{f"{k}_logloss": float("nan") for k in LOGLOSS_METHOD_KEYS}}

    out = {k: roc_auc_score(y_all, np.concatenate(pooled_scores[k])) for k in METHOD_KEYS}

    # Log-loss for the two probability-based methods, reusing the same
    # pooled predictions -- no extra fitting. Unlike AUROC (rank-only), it
    # keeps discriminating between configs whose predictions are perfectly
    # ranked but differ in confidence/calibration.
    eps = 1e-7
    for k in LOGLOSS_METHOD_KEYS:
        p = np.clip(np.concatenate(pooled_scores[k]), eps, 1 - eps)
        out[f"{k}_logloss"] = log_loss(y_all, p, labels=[0, 1])

    return out


# ── REPORT-ONLY: Transfer + Stego, never fed back into selection ──────────

def compute_ood_results(core_meta, core_npz, transfer_meta, transfer_npz,
                         stego_meta, stego_npz, layer, hp: ProbeHParams, seed=0):
    """Score Transfer + Stego with a probe trained on all of Core.

    Call this only to *report* a frozen config's zero-shot performance.
    Do not compare across configs using this to choose a winner -- that is
    exactly the leakage this file exists to avoid.
    """
    scaler, w = train_deception_probe(core_meta, core_npz, layer, hp, seed=seed)
    core_groups = build_groups(core_meta, core_npz, layer, phase="discussion", avg_rounds=True)
    core_rids = sorted(core_groups.keys())
    core_y = _run_labels(core_groups, core_rids)
    core_scores = compute_scores(core_groups, core_rids, scaler, w)

    asym_fs, asym_clf = _train_asymmetry_classifier(core_scores, core_y, hp)
    concat_pca, concat_fs, concat_clf = _train_concat_classifier(core_scores, core_y, hp)

    out = {}

    transfer_groups = build_groups(transfer_meta, transfer_npz, layer, phase="final", avg_rounds=False)
    transfer_rids = sorted(transfer_groups.keys())
    transfer_y = _run_labels(transfer_groups, transfer_rids)
    transfer_sc = [transfer_groups[r]["scenario"] for r in transfer_rids]
    transfer_scores = compute_scores(transfer_groups, transfer_rids, scaler, w)

    asym_proba = asym_clf.predict_proba(asym_fs.transform(transfer_scores["asymmetry_probe"]))[:, 1]
    concat_proba = concat_clf.predict_proba(
        concat_fs.transform(concat_pca.transform(transfer_scores["sorted_concat"])))[:, 1]

    transfer_scenarios = sorted({s for s in transfer_sc if s})
    for sc in transfer_scenarios:
        mask = np.array([s == sc for s in transfer_sc])
        if mask.sum() > 0 and len(np.unique(transfer_y[mask])) >= 2:
            row = {}
            for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
                row[key] = roc_auc_score(transfer_y[mask], transfer_scores[key][mask])
            row["asymmetry_probe"] = roc_auc_score(transfer_y[mask], asym_proba[mask])
            row["sorted_concat"] = roc_auc_score(transfer_y[mask], concat_proba[mask])
            out[sc] = row

    stego_groups = build_groups(stego_meta, stego_npz, layer, phase="final", avg_rounds=False)
    stego_rids = sorted(stego_groups.keys())
    stego_y = _run_labels(stego_groups, stego_rids)
    if len(np.unique(stego_y)) >= 2 and len(stego_rids) >= 4:
        stego_scores = compute_scores(stego_groups, stego_rids, scaler, w)
        row = {}
        for key in ["peak_suspicion", "deception_split", "activation_alignment"]:
            row[key] = roc_auc_score(stego_y, stego_scores[key])
        stego_asym = asym_clf.predict_proba(asym_fs.transform(stego_scores["asymmetry_probe"]))[:, 1]
        row["asymmetry_probe"] = roc_auc_score(stego_y, stego_asym)
        stego_concat = concat_clf.predict_proba(
            concat_fs.transform(concat_pca.transform(stego_scores["sorted_concat"])))[:, 1]
        row["sorted_concat"] = roc_auc_score(stego_y, stego_concat)
        out["blackjack"] = row

    return out


# ── Top-level experiment runner ────────────────────────────────────────────

def run_experiment(core_data, transfer_data, stego_data, hp: ProbeHParams,
                    layers=None, n_seeds=None, compute_ood=True, compute_heldout=True,
                    pooled=True, progress_desc=None):
    """Run held-out CV (+ optionally OOD) across all layers/seeds for one config.

    heldout_mean/heldout_std are computed the same way reproduce_LR.py's
    "Held-out" row is: per-method mean across (layer, seed), then averaged
    across methods.  ood_mean/ood_std mirror reproduce_LR.py's
    "overall_mean" (excl. Held-out): per (group, method) mean across
    (layer, seed), then averaged across all (group, method) cells -- so
    ood_mean here is directly comparable to numbers already in
    results/*/auroc_scores/*.csv from the old (contaminated) sweep.

    pooled controls held_out_cv_layer's fold-aggregation method (see its
    docstring); leave it at the default True unless you're specifically
    comparing against the old per-fold-averaged behaviour.

    compute_heldout=False skips held_out_cv_layer entirely -- useful when
    you only want OOD (e.g. a layer sweep where held-out's 17-domain-fold
    retraining per layer/seed would dominate the runtime for no benefit).
    heldout_mean/heldout_std/heldout_logloss_mean/heldout_logloss_std are
    all None in that case (not NaN, to distinguish "not computed" from
    "computed but undefined").

    progress_desc: if given, shows a tqdm progress bar over the (seed,
    layer) grid for this one config -- this is the loop that actually takes
    the time (full-size runs are ~5 layers x 10 seeds x 17 domains).

    Returns a dict with heldout_mean/heldout_std (report-only under the old
    protocol; see tune_hyperparams.py for why heldout_logloss_mean is the
    actual selection signal now), ood_mean/ood_std (report-only), and the
    raw per-(layer,seed) values for CSV logging.
    """
    layers = layers if layers is not None else DEFAULT_LAYERS
    n_seeds = n_seeds if n_seeds is not None else DEFAULT_N_SEEDS

    core_meta, core_npz = core_data
    transfer_meta, transfer_npz = transfer_data
    stego_meta, stego_npz = stego_data

    heldout_by_method = {k: [] for k in METHOD_KEYS}
    heldout_logloss_by_method = {k: [] for k in LOGLOSS_METHOD_KEYS}
    ood_by_group = defaultdict(lambda: {k: [] for k in METHOD_KEYS})

    grid = [(seed, layer) for seed in range(n_seeds) for layer in layers]
    if progress_desc is not None:
        grid = tqdm(grid, desc=progress_desc, unit="fold-set", leave=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for seed, layer in grid:
            if compute_heldout:
                cv = held_out_cv_layer(core_meta, core_npz, layer, hp, seed=seed, pooled=pooled)
                for k in METHOD_KEYS:
                    if not np.isnan(cv[k]):
                        heldout_by_method[k].append(cv[k])
                if pooled:
                    for k in LOGLOSS_METHOD_KEYS:
                        v = cv.get(f"{k}_logloss", float("nan"))
                        if not np.isnan(v):
                            heldout_logloss_by_method[k].append(v)

            if compute_ood:
                ood = compute_ood_results(core_meta, core_npz, transfer_meta, transfer_npz,
                                           stego_meta, stego_npz, layer, hp, seed=seed)
                for g, row in ood.items():
                    for k in METHOD_KEYS:
                        ood_by_group[g][k].append(row[k])

    heldout_mean = heldout_std = None
    if compute_heldout:
        heldout_cell_means = {
            k: (float(np.mean(v)) if v else float("nan"))
            for k, v in heldout_by_method.items()
        }
        heldout_mean = float(np.nanmean(list(heldout_cell_means.values())))
        heldout_std = float(np.nanstd(list(heldout_cell_means.values())))

    heldout_logloss_mean = heldout_logloss_std = None
    if compute_heldout and pooled:
        logloss_cell_means = [
            float(np.mean(v)) for v in heldout_logloss_by_method.values() if v
        ]
        if logloss_cell_means:
            heldout_logloss_mean = float(np.nanmean(logloss_cell_means))
            heldout_logloss_std = float(np.nanstd(logloss_cell_means))

    ood_mean = ood_std = None
    if compute_ood:
        ood_cell_means = []
        for g, by_method in ood_by_group.items():
            for k in METHOD_KEYS:
                vals = by_method[k]
                if vals:
                    ood_cell_means.append(float(np.mean(vals)))
        if ood_cell_means:
            ood_mean = float(np.nanmean(ood_cell_means))
            ood_std = float(np.nanstd(ood_cell_means))

    return {
        "heldout_mean": heldout_mean,
        "heldout_std": heldout_std,
        "heldout_logloss_mean": heldout_logloss_mean,   # lower is better; None if pooled=False
        "heldout_logloss_std": heldout_logloss_std,
        "ood_mean": ood_mean,
        "ood_std": ood_std,
        "heldout_by_method": heldout_by_method,
        "heldout_logloss_by_method": heldout_logloss_by_method,
        "ood_by_group": dict(ood_by_group),
    }


# ── CLI: run one frozen config by hand ─────────────────────────────────────

def _group_means_stds(result):
    means, stds = {"Held-out": {}}, {"Held-out": {}}
    for k in METHOD_KEYS:
        vals = result["heldout_by_method"][k]
        means["Held-out"][k] = float(np.mean(vals)) if vals else float("nan")
        stds["Held-out"][k] = float(np.std(vals)) if vals else float("nan")
    for g, by_method in result["ood_by_group"].items():
        means[g], stds[g] = {}, {}
        for k in METHOD_KEYS:
            vals = by_method[k]
            means[g][k] = float(np.mean(vals)) if vals else float("nan")
            stds[g][k] = float(np.std(vals)) if vals else float("nan")
    return means, stds


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C-probe", type=float, default=25.0, dest="C_probe")
    parser.add_argument("--max-iter-probe", type=int, default=8, dest="max_iter_probe")
    parser.add_argument("--penalty", choices=["l1", "l2"], default="l2")
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--aug", action="store_true", help="enable Gaussian-noise data augmentation")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=None, dest="n_seeds")
    parser.add_argument("--pooled", dest="pooled", action=argparse.BooleanOptionalAction, default=True,
                         help="pool held-out fold predictions before scoring AUROC (default: True)")
    parser.add_argument("--model", choices=sorted(MODEL_LAYERS), default=DEFAULT_MODEL,
                         help=f"which model's activations to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--out-stem", default="method_comparison_heldout_selected")
    args = parser.parse_args()

    paths = get_paths(args.model)
    hp = ProbeHParams(
        C_probe=args.C_probe, max_iter_probe=args.max_iter_probe,
        penalty=args.penalty, solver=args.solver, aug_enabled=args.aug,
    )
    layers = args.layers or paths.default_layers
    n_seeds = args.n_seeds or DEFAULT_N_SEEDS

    print(f"Loading data (model: {args.model})...")
    core_data = load_data(paths.core_dir)
    transfer_data = load_data(paths.transfer_dir)
    stego_data = load_data(paths.stego_dir)

    print(f"Config: {hp}")
    print(f"Layers: {layers} | Seeds: {n_seeds} | Pooled held-out AUROC: {args.pooled}")
    result = run_experiment(core_data, transfer_data, stego_data, hp,
                             layers=layers, n_seeds=n_seeds, pooled=args.pooled,
                             progress_desc="Held-out CV + OOD")

    print("\n=== SELECTION METRIC (never touches Transfer/Stego) ===")
    print(f"Held-out mean AUROC: {result['heldout_mean']:.4f} +/- {result['heldout_std']:.4f}")
    if result["heldout_logloss_mean"] is not None:
        print(f"Held-out log-loss (asymmetry_probe + sorted_concat, lower=better): "
              f"{result['heldout_logloss_mean']:.4f} +/- {result['heldout_logloss_std']:.4f}")

    print("\n=== REPORT-ONLY (Transfer + Stego; NOT used to pick this config) ===")
    print(f"OOD mean AUROC:      {result['ood_mean']:.4f} +/- {result['ood_std']:.4f}")

    means, stds = _group_means_stds(result)
    group_names = ["Held-out"] + [g for g in means if g != "Held-out"]
    header = f"{'Group':>20}" + "".join(f"  {m:>15}" for m in METHOD_KEYS)
    print(f"\n{header}")
    print("-" * len(header))
    for g in group_names:
        row = f"{g:>20}"
        row += "".join(f"  {means[g][m]:>10.3f}+/-{stds[g][m]:.3f}" for m in METHOD_KEYS)
        print(row)

    paths.csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = paths.csv_dir / f"{args.out_stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_type", "group", "method", "auroc"])
        for g in group_names:
            for m in METHOD_KEYS:
                writer.writerow(["mean", g, m, means[g][m]])
        writer.writerow(["heldout_mean_selection_metric", "", "", result["heldout_mean"]])
        writer.writerow(["heldout_logloss_mean", "", "", result["heldout_logloss_mean"]])
        writer.writerow(["ood_mean_report_only", "", "", result["ood_mean"]])
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
