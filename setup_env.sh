#!/usr/bin/env bash
# One-time setup on Linux (or WSL2). From the project folder:
#   bash setup_env.sh            # NVIDIA GPU: tries CUDA builds cu130, cu128, cu126 in that order
#   CUDA=cu126 bash setup_env.sh # force one build (GTX 10xx / older GPUs: cu126)
#   CPU=1 bash setup_env.sh      # no GPU (testing only)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
CUDA=${CUDA:-auto}

echo "[1/6] Python: $($PY --version)"
echo "[2/6] Creating .venv"
[ -d .venv ] || $PY -m venv .venv
VPY=.venv/bin/python
$VPY -m pip install --upgrade pip wheel

echo "[3/6] Installing PyTorch"
if [ "${CPU:-0}" = "1" ]; then
  $VPY -m pip install torch torchvision
else
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "WARNING: nvidia-smi not found, install the NVIDIA driver"
  if [ "$CUDA" = "auto" ]; then BUILDS="cu130 cu128 cu126"; else BUILDS="$CUDA"; fi
  ok=0
  for b in $BUILDS; do
    echo "Trying PyTorch build $b ..."
    if $VPY -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$b"; then ok=1; break; fi
  done
  [ $ok = 1 ] || { echo "FAILED: no PyTorch CUDA build installed, see https://pytorch.org/get-started/locally/"; exit 1; }
fi

echo "[4/6] Installing project + dev requirements"
$VPY -m pip install -r requirements-dev.txt

echo "[5/6] Local config and git hooks"
[ -f configs/local.yaml ] || { cp configs/local.example.yaml configs/local.yaml; echo "Created configs/local.yaml -> edit your dataset paths"; }
[ -d .git ] && .venv/bin/pre-commit install || true

echo "[6/6] GPU check"
$VPY scripts/check_gpu.py || true
echo "Done. In VS Code: Ctrl+Shift+P > 'Python: Select Interpreter' > .venv"
