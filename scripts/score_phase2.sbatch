#!/usr/bin/env bash
# Score the 800 DreamBooth images. They are already generated.
# From the repo root: sbatch scripts/score_phase2.sbatch
#SBATCH --job-name=db-score
#SBATCH --account=coe
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
# cs002 GPU 0 and cs001 GPUs 1-3 are dead and refuse CUDA work.
#SBATCH --exclude=cs001,cs002
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# DINO and CLIP are already cached. GPU nodes have no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
assert torch.cuda.is_available(), "CUDA is not available"
print("gpu", torch.cuda.get_device_name(0))
torch.zeros(1, device="cuda")
PY

python -u scripts/score_phase2.py
