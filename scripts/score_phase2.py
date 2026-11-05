"""DINO, CLIP-I, and CLIP-T for the Phase 2 DreamBooth images.

Same models and preprocessing as score_phase1.py. CLIP-T uses the prompt without zwx,
which is the exact text Phase 1 was scored against: zwx means nothing to CLIP, and the
same text keeps the two CLIP-T columns comparable.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from configs.subjects import SUBJECTS, TOKEN  # noqa: E402
from score_phase1 import (  # noqa: E402
    CLIP_ID,
    DINO_ID,
    device,
    embed_clip_images,
    embed_clip_texts,
    embed_dino,
    mean_vs_reals,
    real_paths,
    round6,
)

PHASE2_DIR = ROOT / "phase2" / "samples"
OUT_PATH = ROOT / "phase2" / "metrics.json"


def subject_rows(name):
    folder = PHASE2_DIR / name
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
        if TOKEN not in row["prompt"]:
            raise SystemExit(f"Phase 2 prompt is missing {TOKEN}: {row['prompt']}")
        if TOKEN in row["phase1_prompt"]:
            raise SystemExit(f"scoring prompt still contains {TOKEN}: {row['phase1_prompt']}")
        path = folder / row["file"]
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing image {path}")
        files.append(path)
        prompts.append(row["phase1_prompt"])
    return files, prompts


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
        "phase": 2,
        "method": "dreambooth",
        "num_images": 800,
        "dino_model": DINO_ID,
        "clip_model": CLIP_ID,
        "clip_t_prompt": f"prompt without {TOKEN}, same text as Phase 1",
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
