#!/usr/bin/env bash
# The other seven subjects, after dog already looks right.
# Same settings as phase2/train.sh. One A100, one subject after another.
#SBATCH --job-name=db-rest
#SBATCH --account=coe
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
# cs002 GPU 0 and cs001 GPUs 1-3 are dead and refuse CUDA work.
#SBATCH --exclude=cs001,cs002
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
assert torch.cuda.is_available(), "CUDA is not available"
print("gpu", torch.cuda.get_device_name(0))
torch.zeros(1, device="cuda")
PY

for subject in dog2 cat cat2 backpack clock teapot colorful_sneaker; do
  echo "===== ${subject} ====="
  SUBJECT="${subject}" bash phase2/train.sh
done
