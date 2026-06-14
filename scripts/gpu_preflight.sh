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
    echo "  PYTHON=/path/to/python bash scripts/gpu_preflight.sh" >&2
    exit 127
  fi
fi

echo "[1/5] Python environment"
echo "Using Python: $PYTHON_BIN"
if command -v conda >/dev/null 2>&1; then
  echo
  echo "Conda environments:"
  conda env list || true
fi
echo
"$PYTHON_BIN" - <<'PY'
import sys
print(f"python={sys.version.split()[0]}")
print(f"executable={sys.executable}")
PY

echo
echo "[2/5] NVIDIA device nodes"
if compgen -G "/dev/nvidia*" >/dev/null; then
  ls -l /dev/nvidia*
else
  echo "No /dev/nvidia* device nodes are visible in this shell."
fi

echo
echo "[3/5] NVIDIA kernel driver information"
if [[ -r /proc/driver/nvidia/version ]]; then
  cat /proc/driver/nvidia/version
else
  echo "/proc/driver/nvidia/version is not readable."
fi

if compgen -G "/proc/driver/nvidia/gpus/*/information" >/dev/null; then
  echo
  for info in /proc/driver/nvidia/gpus/*/information; do
    echo "$info"
    sed -n '1,80p' "$info"
  done
else
  echo "No NVIDIA GPU information files are visible under /proc/driver/nvidia/gpus."
fi

echo
echo "[4/5] nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo
    echo "nvidia-smi failed in this shell."
  fi
else
  echo "nvidia-smi is not on PATH."
fi

echo
echo "[5/5] PyTorch CUDA check"
if ! "$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch import failed. Install training dependencies first:\n"
        "  python -m pip install -e '.[train,dev]'\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
cuda_available = torch.cuda.is_available()
print(f"cuda_available={cuda_available}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if cuda_available:
    print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    x = torch.ones((1024, 1024), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"cuda_matmul_sum={float(y.sum().item())}")
else:
    raise SystemExit(11)
PY
then
  echo
  echo "GPU preflight failed."
  echo
  echo "What to check next:"
  echo "  1. Run this same command from a normal terminal, not a sandboxed agent session:"
  echo "       PYTHON=$PYTHON_BIN bash scripts/gpu_preflight.sh"
  echo "  2. If the normal terminal can see /dev/nvidia* and torch CUDA works, start training there:"
  echo "       PYTHON=$PYTHON_BIN bash scripts/train_policy.sh"
  echo "  3. If the normal terminal also fails, fix the NVIDIA driver/device nodes first, then reboot if needed."
  exit 1
fi

echo
echo "GPU preflight passed."
echo "Next command:"
echo "  PYTHON=$PYTHON_BIN bash scripts/train_policy.sh"
