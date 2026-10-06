#!/usr/bin/env python3
"""
NARCBench Transfer v2 — gen-time activation extraction.

Standalone script. Walks narcbench_transfer_v2/raw/<family>/<run>/ and, for every
agent message in every run, hooks the hidden state at the final token of the
agent's response at layers 20 through 40 (inclusive) of Qwen3-32B-AWQ.

Inputs
------
raw/<family>/<scenario>__<mode>/
    run_config.json
    agent_prompts.json
    results.json

Outputs
-------
activations/qwen3_32b/
    activations_gen.npz      # arrays layer_20 … layer_40, each (N, 5120) float32
    metadata_gen.json        # N row-aligned records

One sample per agent message:
  - Private round (collusion runs only): 1 sample per colluder per private turn
  - Public discussion:                   1 sample per agent per discussion turn
  - Final summary:                       1 sample per agent

Usage
-----
  cd narcbench_transfer_v2/
  python extract_activations.py

  # custom paths:
  python extract_activations.py --runs-dir raw --output-dir activations/qwen3_32b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Compatibility shim: AutoAWQ 0.2.9 imports PytorchGELUTanh from
# transformers.activations, which was removed in transformers >=4.56. Restore
# the symbol before any AWQ code runs.
# -----------------------------------------------------------------------------
import transformers.activations as _ta
if not hasattr(_ta, "PytorchGELUTanh"):
    class _PytorchGELUTanh(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(x, approximate="tanh")
    _ta.PytorchGELUTanh = _PytorchGELUTanh

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_ROOT, ACTIVATIONS_ROOT

# -----------------------------------------------------------------------------
# Fixed extraction parameters
# -----------------------------------------------------------------------------
MODEL_ID         = "Qwen/Qwen3-32B-AWQ"
TARGET_LAYERS    = list(range(20, 41))       # 20..40 inclusive (21 layers)
CHECKPOINT_EVERY = 100

DEFAULT_RUNS  = RAW_ROOT / "transfer"
DEFAULT_OUT   = ACTIVATIONS_ROOT / "qwen3_32b" / "transfer"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _strip_name_prefix(msg: str) -> str:
    """Remove '[Agent Name]: ' prefix from a stored message."""
    return re.sub(r"^\[[^\]]+\]:\s*", "", msg)


def load_samples(runs_dir: Path) -> list[dict]:
    """Walk raw/<family>/<run>/ and emit one sample per agent-message."""
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    print(f"Found {len(run_dirs)} run directories in {runs_dir}")

    samples: list[dict] = []
    missing: list[str] = []

    for run_dir in run_dirs:
        config_path  = run_dir / "run_config.json"
        prompts_path = run_dir / "agent_prompts.json"
        results_path = run_dir / "results.json"

        if not all(p.exists() for p in (config_path, prompts_path, results_path)):
            missing.append(run_dir.name)
            continue

        cfg     = json.loads(config_path.read_text())
        prompts = json.loads(prompts_path.read_text())
        results = json.loads(results_path.read_text())

        private_responses = [_strip_name_prefix(m) for m in results.get("private_messages", [])]
        public_responses  = [_strip_name_prefix(m) for m in results.get("public_messages",  [])]
        final_texts       = results.get("final_texts", {})

        private_idx = 0
        public_idx  = 0

        for entry in prompts:
            phase   = entry["phase"]
            channel = entry["channel"]
            agent   = entry["agent_name"]

            # Match the stored response to the prompt turn. channel is the
            # stable discriminator for the private round (the phase label was
            # inconsistent in earlier generator versions).
            if channel == "private":
                response = private_responses[private_idx] if private_idx < len(private_responses) else ""
                private_idx += 1
            elif phase == "discussion" and channel == "public":
                response = public_responses[public_idx] if public_idx < len(public_responses) else ""
                public_idx += 1
            elif phase == "final":
                response = final_texts.get(agent, "")
            else:
                response = ""

            samples.append({
                "run_id":       cfg["run_id"],
                "seed":         cfg.get("seed", 0),
                "scenario_id":  cfg["scenario_id"],
                "family":       cfg.get("scenario_type", run_dir.parent.name),
                "mode":         cfg["mode"],
                "agent_name":   agent,
                "role":         entry["role"],
                "phase":        phase,
                "channel":      channel,
                "round":        entry["round"],
                "system_prompt": entry["system_prompt"],
                "user_prompt":   entry["user_prompt"],
                "response":      response,
            })

    if missing:
        print(f"  WARNING: {len(missing)} run dir(s) missing files, skipped: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    print(f"Loaded {len(samples)} samples")
    _print_distribution(samples)
    return samples


def _print_distribution(samples: list[dict]) -> None:
    counts: dict[str, int] = {}
    for s in samples:
        key = f"{s['mode']:10s}  {s['role']:8s}  {s['channel']:8s}  phase={s['phase']}"
        counts[key] = counts.get(key, 0) + 1
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model(model_id: str):
    print(f"Loading tokenizer: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except (ValueError, ImportError):
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    print(f"Loading model:     {model_id} (AutoAWQ, device_map=auto)")
    wrapper = AutoAWQForCausalLM.from_quantized(
        model_id,
        device_map="auto",
        fuse_layers=False,
        trust_remote_code=True,
    )
    model = wrapper.model
    model.eval()
    print("Model ready.")
    return model, tokenizer


# -----------------------------------------------------------------------------
# Activation hooks
# -----------------------------------------------------------------------------

class ActivationHooks:
    """Capture last-token hidden state at each target layer via forward hooks."""

    def __init__(self, model, target_layers: list[int]):
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks = []
        for idx in target_layers:
            hook = model.model.layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, idx: int):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden shape: (batch=1, seq_len, hidden_dim) -> last token, float32
            self.captured[idx] = hidden[0, -1, :].float().cpu()
        return hook_fn

    def get_and_clear(self) -> dict[int, torch.Tensor]:
        result = dict(self.captured)
        self.captured.clear()
        return result

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------

def _build_gen_text(tokenizer, sample: dict) -> str:
    """Reconstruct prompt+response as the model saw it during generation."""
    messages = [
        {"role": "system",    "content": sample["system_prompt"]},
        {"role": "user",      "content": sample["user_prompt"]},
        {"role": "assistant", "content": sample["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        chat_template_kwargs={"enable_thinking": False},
    )


def extract(
    model,
    tokenizer,
    samples: list[dict],
    target_layers: list[int],
    output_dir: Path,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> tuple[dict[int, np.ndarray], list[dict]]:

    ckpt_npz  = output_dir / "_checkpoint_gen.npz"
    ckpt_meta = output_dir / "_checkpoint_gen_meta.json"
    start_idx = 0

    if ckpt_npz.exists() and ckpt_meta.exists():
        print("Resuming from checkpoint...")
        ckpt_data = np.load(ckpt_npz)
        ckpt_info = json.loads(ckpt_meta.read_text())
        start_idx = ckpt_info["next_idx"]
        activations = {
            layer: list(ckpt_data[f"layer_{layer}"][:start_idx])
            for layer in target_layers
        }
        metadata = ckpt_info["metadata"]
        print(f"Resumed at sample {start_idx}/{len(samples)}")
    else:
        activations = {layer: [] for layer in target_layers}
        metadata = []

    hooks = ActivationHooks(model, target_layers)
    hidden_dim = model.config.hidden_size

    try:
        iterator = tqdm(
            range(start_idx, len(samples)),
            initial=start_idx,
            total=len(samples),
            desc="Extracting (gen)",
        )
        for i in iterator:
            sample = samples[i]

            if not sample["response"]:
                n_tokens = 0
                for layer in target_layers:
                    activations[layer].append(np.zeros(hidden_dim, dtype=np.float32))
            else:
                text = _build_gen_text(tokenizer, sample)
                inputs = tokenizer(text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model.device)
                n_tokens = int(input_ids.shape[1])
                with torch.no_grad():
                    model(input_ids)
                captured = hooks.get_and_clear()
                for layer in target_layers:
                    activations[layer].append(captured[layer].numpy())

            metadata.append({
                "run_id":       sample["run_id"],
                "seed":         sample["seed"],
                "scenario_id":  sample["scenario_id"],
                "family":       sample["family"],
                "mode":         sample["mode"],
                "agent_name":   sample["agent_name"],
                "role":         sample["role"],
                "phase":        sample["phase"],
                "channel":      sample["channel"],
                "round":        sample["round"],
                "n_tokens":     n_tokens,
            })

            if (i + 1) % checkpoint_every == 0:
                _save_checkpoint(activations, metadata, target_layers, i + 1, output_dir)
    finally:
        hooks.remove()

    stacked = {layer: np.stack(activations[layer]) for layer in target_layers}
    return stacked, metadata


def _save_checkpoint(activations, metadata, target_layers, next_idx, output_dir):
    arrays = {f"layer_{l}": np.stack(activations[l]) for l in target_layers}
    (output_dir / "_checkpoint_gen.npz").parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "_checkpoint_gen.npz", **arrays)
    (output_dir / "_checkpoint_gen_meta.json").write_text(
        json.dumps({"next_idx": next_idx, "metadata": metadata})
    )


def save_results(activation_arrays, metadata, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "activations_gen.npz"
    np.savez_compressed(
        npz_path, **{f"layer_{l}": arr for l, arr in activation_arrays.items()}
    )
    total_gb = sum(a.nbytes for a in activation_arrays.values()) / 1e9
    first = next(iter(activation_arrays.values()))
    print(f"Saved {npz_path.name}  |  "
          f"{first.shape[0]} samples × {first.shape[1]} dim × "
          f"{len(activation_arrays)} layers  |  {total_gb:.2f} GB")

    meta_path = output_dir / "metadata_gen.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {meta_path.name}  ({len(metadata)} entries)")

    for f in (output_dir / "_checkpoint_gen.npz",
              output_dir / "_checkpoint_gen_meta.json"):
        if f.exists():
            f.unlink()

    counts: dict[str, int] = {}
    for m in metadata:
        key = f"{m['mode']:10s}  {m['role']:8s}  {m['channel']:8s}  phase={m['phase']}"
        counts[key] = counts.get(key, 0) + 1
    print("\nDistribution:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir",   type=Path, default=DEFAULT_RUNS,
                        help=f"Path to raw/ (default: {DEFAULT_RUNS})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--model",      type=str,  default=MODEL_ID,
                        help=f"HuggingFace model id (default: {MODEL_ID})")
    args = parser.parse_args()

    print("=" * 60)
    print("NARCBench Transfer v2 — activation extraction")
    print("=" * 60)
    print(f"Runs dir : {args.runs_dir}")
    print(f"Output   : {args.output_dir}")
    print(f"Layers   : {TARGET_LAYERS[0]}..{TARGET_LAYERS[-1]} inclusive "
          f"({len(TARGET_LAYERS)} layers)")
    print(f"Phase    : gen-time only")
    print()

    if not args.runs_dir.exists():
        print(f"ERROR: runs dir not found: {args.runs_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.runs_dir)
    if not samples:
        print("ERROR: no samples loaded", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load_model(args.model)

    t0 = time.time()
    activations, metadata = extract(model, tokenizer, samples, TARGET_LAYERS, args.output_dir)
    elapsed = time.time() - t0
    print(f"\nExtraction finished in {elapsed/60:.1f} min")

    save_results(activations, metadata, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
