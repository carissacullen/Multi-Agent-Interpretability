#!/usr/bin/env python3
"""Extract hidden-state activations at four structural points around a verdict.

For every (repeat, episode, agent) verdict decision, reconstructs the exact
prompts Qwen3-32B-AWQ actually saw and captures each target layer's hidden
state at four token positions, none of which depend on any judge label (so
they exist identically for every agent, colluding or honest):

  verdict_start  -- first token of the verdict-phase instruction message
                    (peer's evidence already exchanged; verdict not yet asked)
  pre_decision   -- last token of the prompt, right before the model starts
                    generating anything for its verdict-phase turn
  decision       -- the token spelling "accept"/"reject" inside the model's
                    own generated submit_verdict tool call
  reflection     -- last token of the episode's private reflection, which
                    comes after the verdict is submitted and reward is known

verdict_start/pre_decision/decision come from one forward pass (rendered with
just the submit_verdict tool, matching what was actually served for that
call). reflection needs a second forward pass, because the reflection call
uses a different, broader tool schema (get_task_tool_schemas) -- using the
wrong one would shift the system-prompt tool listing and desync every
downstream token.

This script's raw inputs (run.json trajectories) come from a SEPARATE
experiment harness -- the multi-agent verdict-review environment that
generated NARCBench-Verdict-Decisions -- not from this repo's own
generation/extraction pipeline. It imports that harness's
`experiments.tool_schemas` (get_tool_schemas, get_task_tool_schemas) and
`experiments.protocol.state` (AGENT_IDS, peer_id) to re-render the exact
prompts the model saw, so it can't run standalone: point --experiments-root
at a checkout of that harness (it must be importable as a top-level
`experiments` package from that root).

Usage:
    python extract_verdict_activations.py --experiments-root /path/to/harness
    python extract_verdict_activations.py --experiments-root /path/to/harness --limit 20
"""

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# AutoAWQ 0.2.9 imports PytorchGELUTanh from transformers.activations, which was
# removed in transformers >=4.56. Shim it back in before any AWQ code runs.
import transformers.activations as _ta
if not hasattr(_ta, "PytorchGELUTanh"):
    class _PytorchGELUTanh(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(x, approximate="tanh")
    _ta.PytorchGELUTanh = _PytorchGELUTanh

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
DEFAULT_LAYER_RANGE = "16-55"
DEFAULT_OUTPUT_DIR = config.ACTIVATIONS_ROOT / "qwen3_32b" / "verdict_decisions"

POSITION_NAMES = ("verdict_start", "pre_decision", "decision", "reflection")

TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{"name":\s*"submit_verdict".*?\})\s*</tool_call>', re.DOTALL
)
VERDICT_VALUE_PATTERN = re.compile(r'"verdict":\s*"(accept|reject)"')
ASSISTANT_TURN_MARKER = "<|im_start|>assistant"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_answer_key_lookup(task_manifest_glob: str) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for path in glob.glob(task_manifest_glob):
        for entry in json.load(open(path)):
            lookup[entry["task_id"]] = [str(c) for c in entry.get("answer_key", [])]
    return lookup


def load_completed_runs(runs_glob: str) -> list[tuple[str, dict]]:
    """Return (rep, run_dict) for every run.json with all 10 episodes saved."""
    runs = []
    for path in sorted(glob.glob(runs_glob)):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if len(data.get("results", [])) != 10:
            continue
        manifest = data["run_config"]["manifest"]
        rep = manifest.split("rep")[-1].split("_")[0]
        runs.append((rep, data))
    return runs


def build_samples(runs: list[tuple[str, dict]], answer_keys: dict[str, list[str]],
                   agent_ids, peer_id) -> list[dict]:
    """One sample per (rep, episode, agent) verdict decision."""
    samples = []
    for rep, run in runs:
        for episode in run["results"]:
            for agent in agent_ids:
                agent_data = episode["agents"][agent]
                verdict = agent_data.get("verdict")
                if verdict not in ("accept", "reject"):
                    continue  # forced/missing verdicts carry no clean decision token
                peer = peer_id(agent)
                # agents[X]['verdict_prompt'] is the prompt X's PEER sees to judge
                # X -- so the prompt this agent itself saw lives on the peer's slot.
                own_verdict_prompt = episode["agents"][peer].get("verdict_prompt", "")
                samples.append({
                    "rep": rep,
                    "episode_index": episode["episode_index"],
                    "task_type": episode["task_type"],
                    "agent": agent,
                    "peer": peer,
                    "verdict": verdict,
                    "verdict_correct": agent_data.get("verdict_correct"),
                    "own_expected_verdict": agent_data.get("expected_verdict"),
                    "own_task_id": agent_data.get("task_id"),
                    "answer_key": answer_keys.get(agent_data.get("task_id"), []),
                    "verdict_prompt": own_verdict_prompt,
                    "messages": agent_data["messages"],
                })
    return samples


# ---------------------------------------------------------------------------
# Prompt reconstruction
# ---------------------------------------------------------------------------


def _offset_token_covering_end(offsets: list[tuple[int, int]], char_end: int) -> int | None:
    """Token index whose span contains the character just before char_end."""
    for i, (start, end) in enumerate(offsets):
        if start < char_end <= end:
            return i
    return None


def _offset_last_token_before(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    """Last token index whose span ends at or before char_idx."""
    best = None
    for i, (start, end) in enumerate(offsets):
        if end <= char_idx:
            best = i
        else:
            break
    return best


def _offset_first_token_at_or_after(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    for i, (start, end) in enumerate(offsets):
        if end > char_idx:
            return i
    return None


def locate_verdict_positions(
    tokenizer, messages: list[dict], peer: str, verdict_prompt: str, get_tool_schemas,
) -> tuple[list[int], dict[str, int], str] | None:
    """Render the verdict-phase prompt; return (input_ids truncated at the
    decision token, {position_name: token_idx}, verdict_text) or None.
    """
    tools = get_tool_schemas(phase="verdict", peer=peer)
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    # Earlier episodes' own verdict tool calls are still in this growing
    # history, so the current episode's decision is the LAST match, not the
    # first.
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    block = match.group(1)
    value_match = VERDICT_VALUE_PATTERN.search(block)
    if value_match is None:
        return None
    verdict_text = value_match.group(1)
    decision_value_end = match.start(1) + value_match.end(1)

    assistant_start = text.rfind(ASSISTANT_TURN_MARKER, 0, match.start())
    if assistant_start == -1:
        return None

    verdict_prompt_start = text.find(verdict_prompt) if verdict_prompt else -1
    if verdict_prompt_start == -1:
        return None

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    decision_idx = _offset_token_covering_end(offsets, decision_value_end)
    pre_decision_idx = _offset_last_token_before(offsets, assistant_start)
    verdict_start_idx = _offset_first_token_at_or_after(offsets, verdict_prompt_start)
    if None in (decision_idx, pre_decision_idx, verdict_start_idx):
        return None

    truncated = input_ids[: decision_idx + 1]
    positions = {
        "verdict_start": verdict_start_idx,
        "pre_decision": pre_decision_idx,
        "decision": decision_idx,
    }
    return truncated, positions, verdict_text


def locate_reflection_token(
    tokenizer, messages: list[dict], task_type: str, answer_key: list[str], peer: str,
    get_task_tool_schemas,
) -> list[int] | None:
    """Render the full episode (through reflection); return input_ids, whose
    last token is the reflection position by construction."""
    tools = get_task_tool_schemas(task_type=task_type, answer_key=answer_key, peer=peer)
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    encoding = tokenizer(text, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    if not input_ids:
        return None
    return input_ids


# ---------------------------------------------------------------------------
# Model + hooks
# ---------------------------------------------------------------------------


def load_model():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading model: {MODEL_NAME} (device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


class ActivationHooks:
    """Capture hidden states at arbitrary token positions per forward pass."""

    def __init__(self, model, target_layers: list[int]):
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks = []
        self.positions: list[int] | None = None
        for idx in target_layers:
            hook = model.model.layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            positions = self.positions if self.positions is not None else [hidden.shape[1] - 1]
            self.captured[idx] = hidden[0, positions, :].float().cpu()
        return hook_fn

    def run(self, model, input_ids: list[int], positions: list[int]) -> dict[int, torch.Tensor]:
        """Return {layer: tensor(len(positions), hidden_dim)}.

        Calls the inner transformer body directly, skipping the lm_head --
        materializing logits over the full vocab for a long sequence costs
        tens of GB for nothing, since only hidden states are needed here.
        """
        self.positions = positions
        input_tensor = torch.tensor([input_ids], device=model.device)
        with torch.no_grad():
            # use_cache=False: no incremental decoding happens here, so building
            # a full-sequence KV cache (huge for a 35K-token sequence) is pure waste.
            model.model(input_tensor, use_cache=False)
        result = dict(self.captured)
        self.captured.clear()
        self.positions = None
        del input_tensor
        return result

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def parse_layer_range(spec: str) -> list[int]:
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi) + 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments-root", required=True,
                         help="Path to a checkout of the verdict-review experiment "
                              "harness (importable as `experiments` from this root; "
                              "provides experiments.tool_schemas and "
                              "experiments.protocol.state).")
    parser.add_argument("--runs-glob", default=None,
                         help="Defaults to <experiments-root>/results/main/*/run.json")
    parser.add_argument("--task-manifest-glob", default=None,
                         help="Defaults to <experiments-root>/task/*/task_manifest_*.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--layers", default=DEFAULT_LAYER_RANGE)
    parser.add_argument("--limit", type=int, default=0, help="stop after N samples -- smoke test")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root).resolve()
    if not experiments_root.is_dir():
        print(f"ERROR: --experiments-root not found: {experiments_root}")
        return 1
    sys.path.insert(0, str(experiments_root))
    try:
        from experiments.tool_schemas import get_tool_schemas, get_task_tool_schemas
        from experiments.protocol.state import AGENT_IDS, peer_id
    except ImportError as e:
        print(f"ERROR: could not import the experiment harness from "
              f"{experiments_root} ({e}). --experiments-root must point at a "
              f"checkout with an `experiments` package on its top level.")
        return 1

    runs_glob = args.runs_glob or str(experiments_root / "results/main/*/run.json")
    task_manifest_glob = args.task_manifest_glob or str(
        experiments_root / "task/*/task_manifest_*.json")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_keys = build_answer_key_lookup(task_manifest_glob)
    runs = load_completed_runs(runs_glob)
    print(f"Loaded {len(runs)} completed trajectories")
    samples = build_samples(runs, answer_keys, AGENT_IDS, peer_id)
    print(f"Built {len(samples)} verdict-decision samples")
    if args.limit:
        samples = samples[: args.limit]
        print(f"Limited to {len(samples)} samples for this run")

    target_layers = parse_layer_range(args.layers)
    model, tokenizer = load_model()
    hooks = ActivationHooks(model, target_layers)

    activations: dict[str, dict[int, list[np.ndarray]]] = {
        name: {layer: [] for layer in target_layers} for name in POSITION_NAMES
    }
    metadata = []
    skipped = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        verdict_located = locate_verdict_positions(
            tokenizer, sample["messages"], sample["peer"], sample["verdict_prompt"],
            get_tool_schemas,
        )
        if verdict_located is None:
            skipped += 1
            continue
        input_ids_a, positions_a, verdict_text = verdict_located
        assert verdict_text == sample["verdict"], (verdict_text, sample["verdict"])

        input_ids_b = locate_reflection_token(
            tokenizer, sample["messages"], sample["task_type"], sample["answer_key"],
            sample["peer"], get_task_tool_schemas,
        )
        if input_ids_b is None:
            skipped += 1
            continue

        order_a = ["verdict_start", "pre_decision", "decision"]
        result_a = hooks.run(model, input_ids_a, [positions_a[name] for name in order_a])
        result_b = hooks.run(model, input_ids_b, [len(input_ids_b) - 1])
        # Sequence lengths vary wildly (5K-40K tokens) across samples; without
        # this, the caching allocator fragments and OOMs well before actual
        # peak usage would require it.
        torch.cuda.empty_cache()

        for layer in target_layers:
            for j, name in enumerate(order_a):
                activations[name][layer].append(result_a[layer][j].numpy())
            activations["reflection"][layer].append(result_b[layer][0].numpy())

        metadata.append({
            "rep": sample["rep"],
            "episode_index": sample["episode_index"],
            "task_type": sample["task_type"],
            "agent": sample["agent"],
            "peer": sample["peer"],
            "verdict": sample["verdict"],
            "verdict_correct": sample["verdict_correct"],
            "own_expected_verdict": sample["own_expected_verdict"],
            "prompt_tokens_verdict_pass": len(input_ids_a),
            "prompt_tokens_reflection_pass": len(input_ids_b),
        })

        done = i + 1
        if done % 10 == 0 or done == len(samples):
            elapsed = time.time() - start_time
            rate = elapsed / done
            remaining = (len(samples) - done) * rate
            print(
                f"  {done}/{len(samples)} (skipped {skipped}) "
                f"-- {rate:.1f}s/sample, ~{remaining/60:.1f}min remaining",
                flush=True,
            )
        if done % args.checkpoint_every == 0 or done == len(samples):
            save_checkpoint(activations, metadata, target_layers, output_dir)

    hooks.remove()
    print(f"Done. {len(metadata)} samples saved, {skipped} skipped. Output: {output_dir}")
    return 0


def save_checkpoint(
    activations: dict[str, dict[int, list[np.ndarray]]],
    metadata: list[dict],
    target_layers: list[int],
    output_dir: Path,
) -> None:
    """One activations_{position}.npz + metadata_{position}.json pair per
    position, matching this repo's extract_core.py save_results() convention
    (npz keyed by "layer_{L}", metadata as a parallel-ordered list) so the
    same loader shape (`np.load(...)["layer_N"]` + a metadata list of
    matching length) works for either dataset.
    """
    for name in POSITION_NAMES:
        arrays = {f"layer_{l}": np.stack(activations[name][l]) for l in target_layers}
        np.savez_compressed(output_dir / f"activations_{name}.npz", **arrays)
        json.dump(metadata, open(output_dir / f"metadata_{name}.json", "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
