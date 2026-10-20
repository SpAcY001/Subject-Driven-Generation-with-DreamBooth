#!/usr/bin/env bash
# Score the 800 pretrained images. They are already generated.
# From the repo root: sbatch scripts/score_phase1.sbatch
#SBATCH --job-name=dreambooth-score
#SBATCH --account=coe
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
# cs002 GPU 0 has uncorrectable ECC errors and refuses CUDA work.
#SBATCH --exclude=cs002
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# Weights were downloaded on the login node. GPU nodes have no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
print("gpu", torch.cuda.get_device_name(0))
torch.zeros(1, device="cuda")
PY

python -u scripts/score_phase1.py
