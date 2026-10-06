#!/usr/bin/env bash
# Generate scenarios against an already-running vLLM server.

set -euo pipefail

MODEL="Qwen/Qwen3-32B-AWQ"

SCENARIOS_CORE="data/scenarios/eval_aware/qwen3-32b-awq/core"
SCENARIOS_TRANSFER="data/scenarios/eval_aware/qwen3-32b-awq/transfer"
SCENARIOS_STEGO="data/scenarios/eval_aware/qwen3-32b-awq/stego"

echo "--- Generating Core scenarios ---"
python generation/core_eval.py --model "$MODEL" --output-dir "$SCENARIOS_CORE"

echo "--- Generating Transfer scenarios ---"
python generation/transfer_eval.py --model "$MODEL" --output-dir "$SCENARIOS_TRANSFER"

echo "--- Generating Stego scenarios ---"
python generation/stego_eval.py --model "$MODEL" --output-dir "$SCENARIOS_STEGO"

echo "=== Generation complete. ==="
