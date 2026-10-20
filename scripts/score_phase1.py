"""DINO, CLIP-I, and CLIP-T for the pretrained Phase 1 images."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.subjects import SUBJECTS  # noqa: E402

PHASE1_DIR = ROOT / "phase1" / "outputs"
REAL_DIR = ROOT / "data" / "references_512"
OUT_PATH = ROOT / "phase1" / "metrics.json"

DINO_ID = "facebook/dino-vits16"
CLIP_ID = "openai/clip-vit-base-patch32"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH = 32


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_rgb(path):
    return Image.open(path).convert("RGB")


def dino_tensor(image):
    # Shorter side to 256, center crop 224, ImageNet normalize.
    image = image.convert("RGB")
    width, height = image.size
    if width < height:
        new_size = (256, round(height * 256 / width))
    else:
        new_size = (round(width * 256 / height), 256)
    image = image.resize(new_size, Image.BICUBIC)
    width, height = image.size
    left = (width - 224) // 2
    top = (height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (pixels - mean) / std


def batched(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def mean_vs_reals(generated, reals):
    """Mean cosine similarity between one generated image and the real photos."""
    generated = F.normalize(generated.float(), dim=-1)
    reals = F.normalize(reals.float(), dim=-1)
    return (generated @ reals.T).mean(dim=1)


@torch.inference_mode()
def embed_dino(paths, model, chosen):
    rows = []
    for chunk in batched(paths, BATCH):
        pixels = torch.stack([dino_tensor(load_rgb(path)) for path in chunk]).to(chosen)
        hidden = model(pixel_values=pixels).last_hidden_state
        # CLS token. The pooler weights are not in this checkpoint.
        rows.append(hidden[:, 0].cpu())
    return torch.cat(rows, dim=0)


@torch.inference_mode()
def embed_clip_images(paths, model, processor, chosen):
    rows = []
    for chunk in batched(paths, BATCH):
        images = [load_rgb(path) for path in chunk]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(chosen)
        rows.append(model.get_image_features(pixel_values=pixels).cpu())
    return torch.cat(rows, dim=0)


@torch.inference_mode()
def embed_clip_texts(prompts, model, processor, chosen):
    unique = list(dict.fromkeys(prompts))
    encoded = {}
    for chunk in batched(unique, BATCH):
        tokens = processor(text=chunk, return_tensors="pt", padding=True, truncation=True)
        features = model.get_text_features(
            input_ids=tokens["input_ids"].to(chosen),
            attention_mask=tokens["attention_mask"].to(chosen),
        ).cpu()
        for prompt, row in zip(chunk, features):
            encoded[prompt] = row
    return torch.stack([encoded[prompt] for prompt in prompts], dim=0)


def subject_rows(name):
    folder = PHASE1_DIR / name
    rows = json.loads((folder / "metadata.json").read_text())
    if len(rows) != 100:
        raise SystemExit(f"{name} metadata has {len(rows)} rows, expected 100")
    files = []
    prompts = []
    seen = set()
    for row in rows:
        if row["file"] in seen:
            raise SystemExit(f"duplicate file in {name}: {row['file']}")
        seen.add(row["file"])
        if "zwx" in row["prompt"]:
            raise SystemExit(f"Phase 1 prompt contains zwx: {row['prompt']}")
        path = folder / row["file"]
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing image {path}")
        files.append(path)
        prompts.append(row["prompt"])
    return files, prompts


def real_paths(name):
    folder = REAL_DIR / name
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise SystemExit(f"no real photos in {folder}")
    return paths


def round6(value):
    return round(float(value), 6)


def main():
    chosen = device()
    print(f"device={chosen}")
    dino = AutoModel.from_pretrained(DINO_ID).to(chosen).eval()
    clip = CLIPModel.from_pretrained(CLIP_ID).to(chosen).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    per_subject = {}
    dino_scores = []
    clip_i_scores = []
    clip_t_scores = []

    for subject in SUBJECTS:
        name = subject["name"]
        generated_paths, prompts = subject_rows(name)
        reals = real_paths(name)

        dino_gen = embed_dino(generated_paths, dino, chosen)
        dino_real = embed_dino(reals, dino, chosen)
        dino_per_image = mean_vs_reals(dino_gen, dino_real)

        clip_gen = embed_clip_images(generated_paths, clip, processor, chosen)
        clip_real = embed_clip_images(reals, clip, processor, chosen)
        clip_i_per_image = mean_vs_reals(clip_gen, clip_real)

        clip_text = embed_clip_texts(prompts, clip, processor, chosen)
        clip_t_per_image = F.cosine_similarity(clip_gen.float(), clip_text.float(), dim=-1)

        per_subject[name] = {
            "num_images": len(generated_paths),
            "num_real": len(reals),
            "dino": round6(dino_per_image.mean()),
            "clip_i": round6(clip_i_per_image.mean()),
            "clip_t": round6(clip_t_per_image.mean()),
        }
        dino_scores.append(dino_per_image)
        clip_i_scores.append(clip_i_per_image)
        clip_t_scores.append(clip_t_per_image)
        print(
            f"{name}  real={len(reals)}  "
            f"DINO={per_subject[name]['dino']:.4f}  "
            f"CLIP-I={per_subject[name]['clip_i']:.4f}  "
            f"CLIP-T={per_subject[name]['clip_t']:.4f}"
        )

    dino_all = torch.cat(dino_scores)
    clip_i_all = torch.cat(clip_i_scores)
    clip_t_all = torch.cat(clip_t_scores)
    if dino_all.shape[0] != 800:
        raise SystemExit(f"expected 800 images, got {dino_all.shape[0]}")

    metrics = {
        "phase": 1,
        "method": "pretrained",
        "num_images": 800,
        "dino_model": DINO_ID,
        "clip_model": CLIP_ID,
        "dino": round6(dino_all.mean()),
        "clip_i": round6(clip_i_all.mean()),
        "clip_t": round6(clip_t_all.mean()),
        "subjects": per_subject,
    }
    OUT_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"all 800  DINO={metrics['dino']:.4f}  "
        f"CLIP-I={metrics['clip_i']:.4f}  CLIP-T={metrics['clip_t']:.4f}"
    )
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
