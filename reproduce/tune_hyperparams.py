"""
Automates the hyperparameter search for the LR deception probe.

Covers the same search space as the paper's staged search, but REORDERED:
    1. max_iter            (1000, 50, 16, 8, 4)
    2. penalty + solver   (L2+lbfgs, L2+liblinear, L1+saga)
    3. C                  (15, 20, 25, 30, 35)
    4. data augmentation   (off, on -- 4 noisy copies, std=0.05)

each stage carrying forward the previous stage's winner.

The paper tested penalty+solver first, then C, then max_iter, then
augmentation. That order is deliberately NOT replicated here: max_iter is
tuned first (still starting from the same L2/LIBLINEAR/C=25 baseline) so
that, by the time L1/SAGA is evaluated in the solver stage, it inherits
whatever max_iter that stage already picked -- expected to be small (4-16,
consistent with the paper's own finding and our diagnostic), since SAGA is
otherwise painfully slow to (not) converge at max_iter=1000 on this
high-dimensional data (confirmed by an earlier run that had to be killed
after 11 minutes stuck on exactly that). If max_iter tuning doesn't land on
something small, this reordering buys nothing and Stage 2 will be just as
slow on L1/SAGA as before.

FIX #1 vs the original tuning run: every stage below selects its winner
using ONLY a Held-out metric computed on NARCBench-Core (leave-domain-out
CV, from reproduce_LR_heldout.run_experiment). NARCBench-Transfer and
NARCBench-Stego ("OOD mean AUROC") are still computed once for the final
frozen config and written to the audit-trail CSV for comparison against the
old protocol, but the selection logic (see `_selection_key` below) never
reads it during the search.

FIX #2: the selection metric is Held-out AUROC (higher is better), with
Held-out log-loss (lower is better) as a TIE-BREAK. A quick diagnostic
(test_pooling.py) showed Held-out AUROC ceilings near 1.0 for almost every
config on this data -- it's a rank-only metric, and the probe's direction
barely rotates with C on data this separable, so AUROC alone often can't
tell configs apart (multiple C values produced bit-identical AUROC).
Log-loss is sensitive to confidence/calibration, not just ranking, and
showed ~26x more spread across the same test configs -- so it's used to
break exact (or effectively exact) AUROC ties rather than as the primary
signal. `_selection_key` returns (-auroc, logloss) and every candidate list
is sorted with `min(...)`, so higher AUROC always wins first; log-loss only
decides among configs whose Held-out AUROC doesn't distinguish them.

Usage:
    python tune_hyperparams.py                 # full search on Qwen3-32B (default): 10 seeds x 5 layers per candidate
    python tune_hyperparams.py --quick         # 2 seeds x 2 layers, fast smoke test of the search itself
    python tune_hyperparams.py --model gpt_oss_20b   # tune on gpt_oss_20b instead

Results (audit-trail CSV + frozen best-config JSON) are saved under
results/<model>/auroc_scores/rebuttals/, not directly in auroc_scores/, so
repeated runs don't overwrite the other reproduce_*.py output files -- or
each other's -- sitting in that directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_LR_heldout import (
    DEFAULT_N_SEEDS,
    MODEL_LAYERS,
    ProbeHParams,
    get_paths,
    load_data,
    run_experiment,
)

from tqdm import tqdm

# This script's own default -- matches reproduce_LR_heldout.py's
# DEFAULT_MODEL. Pass --model gpt_oss_20b explicitly to tune on that instead.
TUNE_DEFAULT_MODEL = "qwen3_32b"

# Results land under <csv_dir>/rebuttals/ (not directly in auroc_scores/) so
# repeated tuning runs don't overwrite the other reproduce_*.py output files
# -- or each other's -- sitting in that directory.
OUTPUT_SUBDIR = "rebuttals"

# ── Search space (matches the paper's description verbatim) ───────────────

SOLVER_CANDIDATES = [("l2", "lbfgs"), ("l2", "liblinear"), ("l1", "saga")]
C_CANDIDATES = [15, 20, 25, 30, 35]
MAX_ITER_CANDIDATES = [1000, 50, 16, 8, 4]
AUG_CANDIDATES = [False, True]

TOTAL_CONFIGS = (len(SOLVER_CANDIDATES) + len(C_CANDIDATES)
                  + len(MAX_ITER_CANDIDATES) + len(AUG_CANDIDATES) + 1)  # +1 = final confirmation run

LOG_ROWS = []  # every config tried, for the audit-trail CSV


def _selection_key(result):
    """Sort key for min(...): lower is better, compared lexicographically.

    Primary: -Held-out AUROC (so higher AUROC sorts first). Secondary
    (tie-break): Held-out log-loss (lower is better), which only ever
    matters when AUROC doesn't distinguish two configs -- common on this
    data, where AUROC often ceilings and ties exactly across candidates.

    Missing/NaN values (e.g. a fold with no valid domains) sort last on
    that component so they can never win by default.
    """
    auroc = result.get("heldout_mean")
    auroc_key = -auroc if auroc is not None and auroc == auroc else float("inf")

    ll = result.get("heldout_logloss_mean")
    ll_key = ll if ll is not None and ll == ll else float("inf")

    return (auroc_key, ll_key)


def _run(data, hp, layers, n_seeds, compute_ood=False, progress_desc=None):
    """Run one config. compute_ood defaults to False: during the search we
    must not even calculate Transfer/Stego scores, let alone look at them --
    otherwise a human skimming the log could still be swayed by a number
    that was never meant to influence selection. It's switched on exactly
    once, for the final frozen config, at the bottom of main().
    """
    core_data, transfer_data, stego_data = data
    return run_experiment(core_data, transfer_data, stego_data, hp,
                           layers=layers, n_seeds=n_seeds, compute_ood=compute_ood,
                           progress_desc=progress_desc)


def _log(stage, hp: ProbeHParams, result, selected):
    LOG_ROWS.append({
        "stage": stage,
        "penalty": hp.penalty,
        "solver": hp.solver,
        "C_probe": hp.C_probe,
        "max_iter_probe": hp.max_iter_probe,
        "aug_enabled": hp.aug_enabled,
        "heldout_logloss_mean": _fmt(result.get("heldout_logloss_mean")),  # <- selection metric
        "heldout_logloss_std": _fmt(result.get("heldout_logloss_std")),
        "heldout_mean_auroc": result["heldout_mean"],   # report-only now
        "heldout_std_auroc": result["heldout_std"],
        "ood_mean": _fmt(result["ood_mean"]),
        "ood_std": _fmt(result["ood_std"]),
        "selected": selected,
    })


def _fmt(v):
    return "" if v is None else v


def sweep_stage(stage_name, label_fn, base_hp, param_setter, candidates, data, layers, n_seeds,
                overall_pbar):
    """Try each candidate, log every result, return the hp with the best
    Held-out AUROC (log-loss breaking ties -- see _selection_key).

    Only Held-out CV is computed here (compute_ood=False) -- Transfer/Stego
    scores for individual candidates are never calculated, not just unused.

    param_setter(base_hp, candidate) -> new ProbeHParams
    label_fn(candidate) -> short string for the progress line
    """
    print(f"\n[{stage_name}]")
    hps = [param_setter(base_hp, c) for c in candidates]
    results = []
    for hp, cand in zip(hps, candidates):
        t0 = time.time()
        label = label_fn(cand)
        result = _run(data, hp, layers, n_seeds,
                       progress_desc=f"{stage_name} | {label}")
        ll = result.get("heldout_logloss_mean")
        ll_str = f"{ll:.4f}" if ll is not None else "n/a"
        tqdm.write(f"  {label:<28} held-out AUROC={result['heldout_mean']:.4f}  "
                   f"(held-out logloss={ll_str}, tie-break)  "
                   f"[{time.time() - t0:.1f}s]")
        results.append(result)
        overall_pbar.update(1)

    best_i = min(range(len(hps)), key=lambda i: _selection_key(results[i]))
    for i, (hp, result) in enumerate(zip(hps, results)):
        _log(stage_name, hp, result, selected=(i == best_i))

    best_auroc = results[best_i]["heldout_mean"]
    best_ll = results[best_i].get("heldout_logloss_mean")
    best_ll_str = f"{best_ll:.4f}" if best_ll is not None else "n/a"
    tied = sum(1 for r in results if r["heldout_mean"] == best_auroc) > 1
    tie_note = " [tie-broken by log-loss]" if tied else ""
    print(f"  -> selected {label_fn(candidates[best_i])} "
          f"(held-out AUROC={best_auroc:.4f}, logloss={best_ll_str}){tie_note}")
    return hps[best_i]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                         help="2 seeds x 2 layers, for a fast smoke test of the search itself")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=None, dest="n_seeds")
    parser.add_argument("--model", choices=sorted(MODEL_LAYERS), default=TUNE_DEFAULT_MODEL,
                         help=f"which model's activations to tune on (default: {TUNE_DEFAULT_MODEL})")
    parser.add_argument("--out", default="tuning_log_heldout.csv")
    args = parser.parse_args()

    paths = get_paths(args.model)
    layers = args.layers or (paths.default_layers[:2] if args.quick else paths.default_layers)
    n_seeds = args.n_seeds or (2 if args.quick else DEFAULT_N_SEEDS)

    print(f"Loading data (model: {args.model})...")
    data = (load_data(paths.core_dir), load_data(paths.transfer_dir), load_data(paths.stego_dir))

    print(f"Layers: {layers} | Seeds: {n_seeds}")
    print("Selection metric: Held-out AUROC (higher is better), with Held-out log-loss "
          "as a tie-break when AUROC doesn't distinguish two configs. OOD mean AUROC "
          "(Transfer+Stego) is logged for comparison but never used to pick a winner.")

    t_start = time.time()
    base_hp = ProbeHParams(C_probe=25.0, max_iter_probe=1000, penalty="l2",
                            solver="liblinear", aug_enabled=False)

    overall_pbar = tqdm(total=TOTAL_CONFIGS, desc="Overall search progress", unit="config")

    best_hp = sweep_stage(
        "Stage 1/4: max_iter",
        lambda c: f"max_iter={c}",
        base_hp,
        lambda hp, c: replace(hp, max_iter_probe=c),
        MAX_ITER_CANDIDATES, data, layers, n_seeds, overall_pbar)

    print(f"  (max_iter={best_hp.max_iter_probe} carries into Stage 2 -- "
          f"L1/SAGA will only run that many iterations, not 1000)")

    best_hp = sweep_stage(
        "Stage 2/4: penalty + solver",
        lambda c: f"penalty={c[0]} solver={c[1]}",
        best_hp,
        lambda hp, c: replace(hp, penalty=c[0], solver=c[1]),
        SOLVER_CANDIDATES, data, layers, n_seeds, overall_pbar)

    best_hp = sweep_stage(
        "Stage 3/4: C (regularisation strength)",
        lambda c: f"C={c}",
        best_hp,
        lambda hp, c: replace(hp, C_probe=c),
        C_CANDIDATES, data, layers, n_seeds, overall_pbar)

    best_hp = sweep_stage(
        "Stage 4/4: data augmentation",
        lambda c: f"aug={c}",
        best_hp,
        lambda hp, c: replace(hp, aug_enabled=c, aug_noise_std=0.05, aug_n_copies=4),
        AUG_CANDIDATES, data, layers, n_seeds, overall_pbar)

    print(f"\n=== Final frozen config (selected using Held-out AUROC, "
          f"log-loss as tie-break) ===\n{best_hp}")

    # The ONE point in the whole search where OOD is computed at all --
    # for the already-frozen config, purely to report a zero-shot number.
    final_result = _run(data, best_hp, layers, n_seeds, compute_ood=True,
                         progress_desc="Final confirmation run")
    overall_pbar.update(1)
    overall_pbar.close()

    _log("final_frozen_config", best_hp, final_result, selected=True)
    final_ll = final_result.get("heldout_logloss_mean")
    print(f"Held-out mean AUROC (selection metric): "
          f"{final_result['heldout_mean']:.4f} +/- {final_result['heldout_std']:.4f}")
    print(f"Held-out log-loss (tie-break):           "
          f"{final_ll:.4f} +/- {final_result['heldout_logloss_std']:.4f}" if final_ll is not None
          else "Held-out log-loss (tie-break):           n/a")
    print(f"OOD mean AUROC (report-only):           "
          f"{final_result['ood_mean']:.4f} +/- {final_result['ood_std']:.4f}")
    print(f"\nTotal search time: {time.time() - t_start:.1f}s")

    out_dir = paths.csv_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / args.out
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LOG_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(LOG_ROWS)
    print(f"\nFull audit trail (every config tried, held-out log-loss, held-out AUROC, "
          f"and ood, with a 'selected' flag) saved to {log_path}")

    config_path = out_dir / f"best_config_heldout_{args.model}.json"
    with open(config_path, "w") as f:
        json.dump(best_hp.__dict__, f, indent=2)
    print(f"Frozen best config saved to {config_path}")

    aug_flag = " --aug" if best_hp.aug_enabled else ""
    print("\nRun it standalone with:")
    print(f"  python reproduce_LR_heldout.py --model {args.model} --C-probe {best_hp.C_probe} "
          f"--max-iter-probe {best_hp.max_iter_probe} --penalty {best_hp.penalty} "
          f"--solver {best_hp.solver}{aug_flag}")


if __name__ == "__main__":
    main()
