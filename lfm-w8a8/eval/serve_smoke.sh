#!/usr/bin/env bash
# vLLM serve smoke for a quantized LFM2.5-24B-A2B dir.
#
# usage: bash eval/serve_smoke.sh ~/models/LFM2.5-24B-A2B-W8A8
#
# The shared ../../.venv (repo root) has torch/transformers/llmcompressor but
# NOT vllm (see ../../shared/ENVIRONMENT.md), so this looks for a system/venv vllm first
# and otherwise falls back to the docker image. Serve needs both GPUs free.
set -euo pipefail

MODEL="${1:?usage: serve_smoke.sh <model-dir>}"
PORT="${PORT:-8000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TP="${TP:-2}"

if command -v vllm >/dev/null 2>&1; then
  echo "== serving with vllm from PATH =="
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
    --trust-remote-code --served-model-name lfm2 &
elif [ -x "$SCRIPT_DIR/../../.venv/bin/vllm" ]; then
  echo "== serving with ../../.venv/bin/vllm =="
  "$SCRIPT_DIR/../../.venv/bin/vllm" serve "$MODEL" --port "$PORT" \
    --tensor-parallel-size "$TP" --trust-remote-code \
    --served-model-name lfm2 &
elif command -v docker >/dev/null 2>&1; then
  echo "== serving with docker (vllm/vllm-openai) =="
  docker run --rm --gpus all -p "$PORT:8000" \
    -v "$(readlink -f "$MODEL")":/model \
    vllm/vllm-openai:v0.29.0 serve /model --trust-remote-code \
    --tensor-parallel-size "$TP" --served-model-name lfm2 &
else
  echo "error: no vllm found (PATH, ../../.venv/bin/vllm, or docker)" >&2
  echo "  note: the shared .venv deliberately does not install vllm" >&2
  exit 1
fi

SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  sleep 5
done

curl -sf "http://localhost:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2","messages":[{"role":"user","content":"Say OK."}],
       "max_tokens":8,"temperature":0.1,"top_k":50}'
echo
# NOTE: this model always thinks — the chat template opens every assistant
# turn with a reasoning block, so raise max_tokens for a real check and leave
# budget for the trace. Recommended: temperature=0.1, top_k=50,
# repetition_penalty=1.05.
