#!/usr/bin/env bash
# Train one subject.
#   SUBJECT=dog bash phase2/train.sh
# Needs train_dreambooth.py from the Diffusers examples.

set -euo pipefail

if [[ -z "${SUBJECT:-}" ]]; then
  echo "Set SUBJECT to one of: dog dog2 cat cat2 backpack clock teapot colorful_sneaker" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLASS_NAME="$(python3 - <<PY
import sys
sys.path.insert(0, "${ROOT}")
from configs.subjects import subject_by_name
print(subject_by_name("${SUBJECT}")["class_name"])
PY
)"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" \
  --instance_data_dir="data/references_512/${SUBJECT}" \
  --class_data_dir="phase2/class_images/${CLASS_NAME}" \
  --output_dir="phase2/outputs/${SUBJECT}" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of zwx ${CLASS_NAME}" \
  --class_prompt="a photo of a ${CLASS_NAME}" \
  --resolution=512 \
  --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
