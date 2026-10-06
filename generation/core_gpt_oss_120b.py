"""NARCBench-Core generation, gpt-oss-120b variant.

Reuses core.py's scenario bank, agent pool, round structure, labeling, and
file-writing logic entirely unchanged (import core, monkey-patch call_vllm,
delegate to core.main()) -- same pattern as core_natural.py. The only thing
that differs is how the LLM is actually called: gpt-oss-120b is a reasoning
model served via vLLM's native Harmony/gpt-oss support
(`reasoning_parser=openai_gptoss`), not Qwen3, so:

  - Reasoning control is gpt-oss's own `reasoning_effort` chat-template
    variable (low/medium/high, defaults to "medium" -- confirmed directly
    against openai/gpt-oss-120b's tokenizer_config.json chat template), not
    Qwen3's `enable_thinking` boolean core.call_vllm uses. Set to "low"
    here: committee-deliberation responses don't need deep reasoning, and
    low effort keeps generated-token volume (and wall-clock time) down.
  - With reasoning_parser=openai_gptoss configured server-side, vLLM's
    OpenAI-compatible /chat/completions response already splits Harmony
    output into `reasoning_content` and `content` -- message.content is
    already the clean final answer, no manual tag-stripping needed.
  - Falls back to reasoning_content if content comes back empty: the
    Harmony parser has been observed (on this exact server setup) to
    occasionally end in a non-terminal state and return raw/recovered
    output instead of a clean split -- rather than silently returning
    nothing, downstream regex parsing (RECOMMENDATION: ...) gets SOMETHING
    to look at.

Output goes to its own scenario directory (core_gpt_oss_120b, not core/),
since the generated CONVERSATION CONTENT differs by model even though the
scenario prompts and round structure are identical -- writing into the
existing core/ directory would collide run_ids with the original
Qwen3-generated transcripts.

Requires a vLLM server already running gpt-oss-120b with Harmony support,
e.g. (the confirmed-working config from this project's runpod2 setup):
    vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 \\
        --reasoning-parser openai_gptoss --port 8020

Usage:
    python core_gpt_oss_120b.py --test                     # 3 scenarios, smoke test
    python core_gpt_oss_120b.py --base-url http://127.0.0.1:8020/v1
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"
DEFAULT_OUTPUT_DIR = config.RAW_ROOT / "core_gpt_oss_120b"


def call_vllm_gpt_oss(system: str, user: str, base_url: str, model: str,
                       temperature: float, max_tokens: int) -> str:
    """Drop-in replacement for core.call_vllm, gpt-oss-120b Harmony variant."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"reasoning_effort": REASONING_EFFORT},
    }
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=180,  # generous: reasoning models can be slower to first token
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            content = (message.get("reasoning_content") or "").strip()
        return content
    except Exception as e:
        print(f"  [API ERROR] {e}")
        return "[ERROR]"


def _inject_default(flag: str, value) -> None:
    """Insert `flag value` into sys.argv unless the caller already passed
    that flag explicitly -- so core.main()'s own argparse still wins if the
    user overrides it on the command line. Needed because core.py's
    --output-dir default is a literal baked into its own argparse call
    (config.RAW_ROOT / "core"), not a module-level constant we could
    reassign the way MODEL/BASE_URL are.
    """
    if not any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:]):
        sys.argv += [flag, str(value)]


def main():
    core.call_vllm = call_vllm_gpt_oss
    _inject_default("--model", MODEL)
    _inject_default("--output-dir", DEFAULT_OUTPUT_DIR)
    core.main()


if __name__ == "__main__":
    main()
