#!/usr/bin/env bash
# Train one subject.
#   SUBJECT=dog bash phase2/train.sh
# phase2/train_dreambooth.py is the Diffusers v0.39.0 example, unchanged.

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

# model_has_vae() in train_dreambooth.py asks the Hub API unless it gets a local folder,
# and GPU nodes have no internet, so pass the cached snapshot of the same model.
MODEL_PATH="$(python3 - <<PY
import sys
sys.path.insert(0, "${ROOT}")
from huggingface_hub import snapshot_download
from configs.subjects import MODEL_ID
print(snapshot_download(MODEL_ID, local_files_only=True))
PY
)"

# train_dreambooth.py opens every file in the class folder, and generate.py keeps a
# metadata.json there, so training reads a folder of links to the PNGs only.
CLASS_DIR="phase2/class_images/${CLASS_NAME}"
PNG_DIR="phase2/class_png/${CLASS_NAME}"
COUNT="$(find "${CLASS_DIR}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"
if [[ "${COUNT}" -lt 200 ]]; then
  echo "${CLASS_DIR} has ${COUNT} class images, need 200. Run scripts/generate.py --class-images first." >&2
  exit 1
fi
mkdir -p "${PNG_DIR}"
ln -sf "${ROOT}/${CLASS_DIR}"/*.png "${PNG_DIR}/"

# checkpointing_steps is above max_train_steps, so only the final model is saved. A
# mid-run checkpoint also keeps the optimizer state, about 10 GB per subject.
# python -m runs accelerate from the active venv; the cluster module puts its
# accelerate command outside .venv/bin.
python3 -m accelerate.commands.launch --num_processes=1 --mixed_precision=no phase2/train_dreambooth.py \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --instance_data_dir="data/references_512/${SUBJECT}" \
  --class_data_dir="${PNG_DIR}" \
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
  --max_train_steps=800 \
  --checkpointing_steps=100000 \
  --seed=0
