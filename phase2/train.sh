#!/usr/bin/env bash
# Train one subject. Full DreamBooth, prior preservation on.
#   SUBJECT=dog bash phase2/train.sh
# Uses train_dreambooth.py from Diffusers 0.39.

set -euo pipefail

if [[ -z "${SUBJECT:-}" ]]; then
  echo "Set SUBJECT to one of: dog dog2 cat cat2 backpack clock teapot colorful_sneaker" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# GPU nodes have no internet. Read the cached Stable Diffusion snapshot.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
# Accelerate imports DeepSpeed, and that import looks for nvcc.
export CUDA_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"

CLASS_NAME="$(python3 - <<PY
import sys
sys.path.insert(0, "${ROOT}")
from configs.subjects import subject_by_name
print(subject_by_name("${SUBJECT}")["class_name"])
PY
)"

MODEL_DIR="$(python3 - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("sd-legacy/stable-diffusion-v1-5", local_files_only=True))
PY
)"

mkdir -p "phase2/outputs/${SUBJECT}"

# fp16 and gradient checkpointing so this fits on one GPU.
# Text encoder is on so "zwx" stays this dog.
# 800 steps at 5e-6 copied the training photos. 800 steps at 2e-6 still ignored the scene and broke the face.
# 400 steps at 5e-6 is the middle.
# No mid-run checkpoint. 1000 is past the step count. The saved pipeline is what we keep.
python3 -m accelerate.commands.launch \
  --mixed_precision=fp16 \
  --num_processes=1 \
  --num_machines=1 \
  phase2/train_dreambooth.py \
  --pretrained_model_name_or_path="${MODEL_DIR}" \
  --instance_data_dir="data/references_512/${SUBJECT}" \
  --class_data_dir="phase2/class_images/${CLASS_NAME}" \
  --output_dir="phase2/outputs/${SUBJECT}" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --train_text_encoder \
  --instance_prompt="a photo of zwx ${CLASS_NAME}" \
  --class_prompt="a photo of a ${CLASS_NAME}" \
  --resolution=512 \
  --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=400 \
  --checkpointing_steps=1000 \
  --mixed_precision=fp16 \
  --allow_tf32
