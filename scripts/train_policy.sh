#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
else
  PYTHON_BIN=""
  for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "No Python interpreter found. Set PYTHON explicitly, for example:" >&2
    echo "  PYTHON=/path/to/python bash scripts/train_policy.sh" >&2
    exit 127
  fi
fi
TASK_COUNT="${TASK_COUNT:-30}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_PPO="${RUN_PPO:-1}"
RUN_GRPO="${RUN_GRPO:-1}"
EVAL_AGENT="${EVAL_AGENT:-grpo}"
ALLOW_CPU_TRAINING="${ALLOW_CPU_TRAINING:-0}"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/8] Checking Python and PyTorch"
echo "Using Python: $PYTHON_BIN"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi:"
  if ! nvidia-smi; then
    echo "warning: nvidia-smi failed. GPU training will not work until the NVIDIA driver/runtime is fixed." >&2
  fi
else
  echo "warning: nvidia-smi is not on PATH. GPU training requires a working NVIDIA driver." >&2
fi
"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python >= 3.11 is required, got {sys.version.split()[0]}")

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch is required for real policy training. Install with:\n"
        "  python -m pip install -e '.[train,dev]'\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"cuda_available={cuda_available}")
if cuda_available:
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
else:
    import os
    if os.environ.get("ALLOW_CPU_TRAINING") == "1":
        print("warning: CUDA is not available; ALLOW_CPU_TRAINING=1 so continuing on CPU.")
    else:
        raise SystemExit(
            "CUDA is not available. This training script defaults to GPU-only training.\n"
            "Fix the NVIDIA driver/CUDA setup, or set ALLOW_CPU_TRAINING=1 for CPU debugging.\n"
            "If this failure only happens inside a sandboxed agent session, run the same command "
            "from a normal terminal where /dev/nvidia* is visible."
        )
PY

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  echo "[2/8] Creating benchmark tasks"
  "$PYTHON_BIN" -m agentic_code_rl benchmark create --out data/tasks --count "$TASK_COUNT" --overwrite
else
  echo "[2/8] Skipping benchmark creation"
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[3/8] Running tests"
  "$PYTHON_BIN" -m pytest -q
else
  echo "[3/8] Skipping tests"
fi

if [[ "$RUN_SFT" == "1" ]]; then
  echo "[4/8] Training SFT policy"
  "$PYTHON_BIN" -m agentic_code_rl train-sft --config configs/sft.yaml
else
  echo "[4/8] Skipping SFT"
fi

if [[ "$RUN_PPO" == "1" ]]; then
  echo "[5/8] Training PPO policy"
  "$PYTHON_BIN" -m agentic_code_rl train-ppo --config configs/ppo.yaml
else
  echo "[5/8] Skipping PPO"
fi

if [[ "$RUN_GRPO" == "1" ]]; then
  echo "[6/8] Training GRPO policy"
  "$PYTHON_BIN" -m agentic_code_rl train-grpo --config configs/grpo.yaml
else
  echo "[6/8] Skipping GRPO"
fi

echo "[7/8] Evaluating ${EVAL_AGENT}"
"$PYTHON_BIN" -m agentic_code_rl eval --config configs/eval.yaml --agent "$EVAL_AGENT"

echo "[8/8] Writing report"
"$PYTHON_BIN" -m agentic_code_rl report --run runs/latest

echo
echo "Training pipeline complete."
echo "Report: $ROOT_DIR/runs/latest/report.md"
