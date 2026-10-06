"""Shared configuration for NARCBench."""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_ROOT        = ROOT / "data"
RAW_ROOT         = DATA_ROOT / "scenarios"
ACTIVATIONS_ROOT = DATA_ROOT / "activations"
RESULTS_ROOT     = ROOT / "results" / "qwen3_32b"

# ── Model defaults ────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "Qwen/Qwen3-32B-AWQ"
VLLM_BASE_URL    = "http://127.0.0.1:8020/v1"
VLLM_TEMPERATURE = 0.7
VLLM_MAX_TOKENS  = 512

