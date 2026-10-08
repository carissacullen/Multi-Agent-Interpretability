# NARCBench

Code and data for **Detecting Multi-Agent Collusion Through Multi-Agent Interpretability**.

NARCBench is a three-tier benchmark for evaluating multi-agent collusion detection under environment distribution shift. A linear deception probe is trained on committee-deliberation activations (Core) and evaluated zero-shot on structurally novel scenarios (Transfer) and steganographic signalling (Stego).

**Paper**: [arXiv:2604.01151](https://arxiv.org/abs/2604.01151)

**Data**: [HuggingFace dataset](https://huggingface.co/datasets/aaronrose227/narcbench)

## Quick Start

```bash
pip install numpy scikit-learn matplotlib peft
python reproduce/reproduce_LR.py
```

This reproduces Table 1 and Figure 2 from the paper: trains a deception direction on NARCBench-Core and evaluates five detection methods zero-shot on NARCBench-Transfer and NARCBench-Stego, averaged over 10 probe-training seeds with the tuned hyperparameters from the paper's search (see `reproduce/tune_hyperparams.py`). Requires the pre-extracted activation data from the HuggingFace dataset (see Data section below). Output (a figure and a full CSV of per-seed/layer results) is saved to `results/qwen3_32b/figures/` and `results/qwen3_32b/auroc_scores/`.

## Data

The activation data and scenario data are hosted on HuggingFace due to size (~2.6GB). Download them from the HuggingFace dataset link above and place them in `data/`.

## Structure

```
narcbench/
├── README.md
├── config.py                          # Shared config (model, paths, vLLM settings)
├── colours.py                         # Shared colour palettes (teal/red/green) for plots
├── roc_analysis.py                    # Per-tier ROC curves and Recall@1%/5% FPR tables
├── requirements.txt                   # Pinned Python dependencies
├── generate.sh                        # Generate eval-aware Core/Transfer/Stego scenarios (needs running vLLM server)
├── extract.sh                         # Extract activations for the eval-aware scenarios (needs GPU)
├── generation/                        # Scenario generation (requires vLLM server)
│   ├── core.py                        # NARCBench-Core (50 committee deliberation scenarios)
│   ├── transfer.py                    # NARCBench-Transfer (6 structurally distinct types)
│   ├── stego.py                       # NARCBench-Stego (steganographic card counting)
│   ├── core_eval.py                   # Core with an "this is a research evaluation" notice prepended (eval-awareness test)
│   ├── transfer_eval.py               # Transfer, eval-aware variant
│   ├── stego_eval.py                  # Stego, eval-aware variant
│   ├── collusion_instructed_pairs.py  # Contrastive collude/honest pairs (analogue of Instructed-Pairs, Goldowsky-Dill et al.)
│   ├── judge_informed_test.py         # Stego text monitor: judge briefed on the card-counting threat model
│   ├── judge_fewshot_test.py          # Stego text monitor: informed judge plus few-shot examples
│   ├── judge_strongmodel_test.py      # Stego text monitor: stronger hosted judge model via litellm
│   └── verify_ground_truth.py         # Ground truth verification
├── extraction/                        # Activation extraction (requires GPU + model weights)
│   ├── extract_core.py                # Core activation extraction
│   ├── extract_transfer.py            # Transfer activation extraction
│   ├── extract_stego.py               # Stego activation extraction
│   ├── extract_token_traces.py        # Per-token deception traces (projection onto the probe direction)
│   └── extract_verdict_activations.py # Verdict Decisions activation extraction (see below)
└── reproduce/                         # Probe training + evaluation
    ├── reproduce_LR.py                # Reproduces Table 1 and Figure 2 (five-method comparison, logistic regression probe)
    ├── reproduce_LR_heldout.py        # LR variant; hyperparameters chosen using only leave-domain-out Core AUROC
    ├── tune_hyperparams.py            # Staged hyperparameter search for the LR probe (held-out metric only)
    ├── reproduce_LAT.py               # Same comparison with a LAT (PCA-based) direction
    ├── reproduce_diffMeans.py         # Same comparison with a difference-of-means direction
    ├── reproduce_mlp.py               # Same comparison with an MLP probe
    ├── reproduce_multiHead.py         # Same comparison with a multi-head (PyTorch) probe
    ├── reproduce_multiLayer.py        # Same comparison with LR over multiple layers
    ├── reproduce_LR_eval_aware.py     # Core-trained probe, zero-shot on eval-aware Transfer/Stego
    ├── reproduce_LR_core_to_verdict.py        # Core-trained probe, zero-shot on Verdict Decisions
    ├── reproduce_LR_core_to_verdict_no_mutual.py  # Same, excluding mutual-collusion episodes
    ├── reproduce_LR_core_to_exploit_share.py  # Cross-model: Qwen3 Core probe -> gpt-oss-120b Exploit-Share (truncated dims)
    ├── reproduce_LR_verdict_to_exploit_share.py  # Cross-model: Qwen3 Verdict probe -> gpt-oss-120b Exploit-Share
    └── comparison_plot.py             # Bar chart comparing two result CSVs
```

## Pipeline

1. **Generate scenarios** (`generation/`): Requires a running vLLM server with Qwen3-32B-AWQ. Each script generates scenario data (prompts, conversations, outcomes) for one benchmark tier.
2. **Extract activations** (`extraction/`): Loads the model locally and extracts hidden-state activations at specified layers.
3. **Evaluate probes** (`reproduce/reproduce_LR.py`): Trains a deception probe on Core activations and evaluates five probing techniques across all three benchmark tiers.

## Activation Oracle

`query_oracle_from_npz.py` lets you ask natural-language questions about agent activations using the [`adamkarvonen/checkpoints_latentqa_only_Qwen3-32B`](https://huggingface.co/adamkarvonen/checkpoints_latentqa_only_Qwen3-32B) oracle LoRA, reading directly from the pre-extracted `.npz` files without re-running the base model on raw text.

**Interactive mode** — browse samples and type questions at the prompt:

```bash
python query_oracle_from_npz.py
python query_oracle_from_npz.py --split transfer --layer 28
```

**Batch mode** — run one question over all samples and save results to JSON:

```bash
python query_oracle_from_npz.py --batch --question "Is this agent colluding?"
python query_oracle_from_npz.py --batch --split stego --role colluder --output results/oracle_stego.json
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--split` | `core` | Dataset split to load (`core`, `transfer`, `stego`) |
| `--layer` | `30` | Layer index from the npz (oracle trained on layer 32; 30 is closest available) |
| `--role` | — | Filter by agent role (`colluder` or `honest`) |
| `--phase` | — | Filter by phase (`discussion` or `final`) |
| `--run-id` | — | Filter to a specific run |
| `--output` | `results/oracle_from_npz/results.json` | Output path for batch results |
| `--base-model` | `Qwen/Qwen3-32B-AWQ` | Base model ID |
| `--oracle` | `adamkarvonen/checkpoints_latentqa_only_Qwen3-32B` | Oracle LoRA ID |

Requires the base model weights and a GPU with sufficient VRAM (~40GB+).

## Data Format

Each scenario run directory contains:
- `run_config.json` — Scenario setup, agent roles, and coalition membership
- `results.json` — Full conversation transcripts, votes, and outcomes
- `agent_prompts.json` — Exact system and user prompts sent to each agent per round

## Verdict Decisions: emergent collusion transfer

A separate probe-transfer experiment on **Verdict Decisions**, a two-agent (alice/bob) repeated peer-verdict setting where each agent completes a task, then reviews and verdicts (accept/reject) its partner's submission. Unlike NARCBench-Core, collusion here is never instructed — it emerges (or doesn't) from the agents' own incentives, generated by a separate experiment harness (not part of this repo's own `generation/`/`extraction/` pipeline).

Activations were captured at 4 structural points in the verdict-giving process (`pre_decision`, `verdict_start`, `decision`, `reflection`; `data/activations/qwen3_32b/verdict_decisions/`), extracted via `extraction/extract_verdict_activations.py` (requires `--experiments-root` pointing at a checkout of that harness).

`reproduce/reproduce_LR_core_to_verdict.py` trains the same five detection methods on NARCBench-Core and evaluates them zero-shot on all four Verdict Decisions phases, generalized to work on 2-agent episodes; `reproduce_LR_core_to_verdict_no_mutual.py` reruns the same comparison excluding episodes where both agents colluded (a cleaner one-deviant-agent-vs-honest-pair contrast).

## Citation

```bibtex
@article{rose2026narcbench,
  title={Detecting Multi-Agent Collusion Through Multi-Agent Interpretability},
  author={Rose, Aaron and Cullen, Carissa and Abdelnabi, Sahar and Torr, Philip and Kaplowitz, Brandon Gary and Schroeder de Witt, Christian},
  year={2026},
  eprint={2604.01151},
  archivePrefix={arXiv}
}
```

## License

MIT
