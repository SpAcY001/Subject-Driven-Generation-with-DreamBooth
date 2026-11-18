#!/usr/bin/env bash
# One subject.   SUBJECT=dog sbatch phase2/train.sbatch
# H100 nodes were busy, so this asks for an A100.
# cs002 GPU 0 and cs001 GPUs 1-3 are dead and refuse CUDA work.
#SBATCH --job-name=dreambooth
#SBATCH --account=coe
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs001,cs002
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

if [[ -z "${SUBJECT:-}" ]]; then
  echo "Pass SUBJECT when you submit, for example SUBJECT=dog sbatch phase2/train.sbatch" >&2
  exit 1
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
print("gpu", torch.cuda.get_device_name(0))
torch.zeros(1, device="cuda")
PY

echo "training ${SUBJECT}"
bash phase2/train.sh
