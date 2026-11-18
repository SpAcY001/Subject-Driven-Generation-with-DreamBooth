#!/usr/bin/env bash
# Phase 2 images from the fine-tuned models. Does not touch phase1/outputs.
# From the repo root:
#   sbatch scripts/generate_phase2.sbatch --subject dog --prompt-index 2
# Extra args are passed straight to scripts/generate.py.
#SBATCH --job-name=db-gen
#SBATCH --account=coe
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
# cs002 GPU 0 and cs001 GPUs 1-3 are dead and refuse CUDA work.
#SBATCH --exclude=cs001,cs002
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# Weights are already on disk. GPU nodes have no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA is not available"
print("gpu", torch.cuda.get_device_name(0))
torch.zeros(1, device="cuda")
PY

python -u scripts/generate.py --phase2 "$@"
