#!/usr/bin/env bash
# Phase 2: train one subject per array task, then make its 100 Phase 2 images.
# Needs the class images from scripts/generate.sbatch.
# From the repo root: sbatch phase2/train.sbatch
# One subject only:   SUBJECT=dog sbatch --array=0 phase2/train.sbatch
#SBATCH --job-name=dreambooth-train
#SBATCH --account=coe
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
# cs002 GPU 0 has uncorrectable ECC errors and refuses CUDA work. On November 4, four
# of our tasks also got cs001 GPUs that reported CUDA not available.
#SBATCH --exclude=cs001,cs002
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# Weights were downloaded on the login node. GPU nodes have no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
# The cluster torch environment ships DeepSpeed. accelerate imports it during training,
# and DeepSpeed fails at import without CUDA_HOME.
export CUDA_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6

SUBJECTS=(dog dog2 cat cat2 backpack clock teapot colorful_sneaker)
export SUBJECT="${SUBJECT:-${SUBJECTS[${SLURM_ARRAY_TASK_ID}]}}"
echo "subject ${SUBJECT}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
print("gpu", torch.cuda.get_device_name(0))
torch.zeros(1, device="cuda")
PY

# A finished model has model_index.json. Skip training on a rerun.
if [[ ! -f "phase2/outputs/${SUBJECT}/model_index.json" ]]; then
  bash phase2/train.sh
fi
python -u scripts/generate.py --phase2 --subject "${SUBJECT}"
