"""Phase 1 images and class images from pretrained Stable Diffusion."""

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.subjects import (  # noqa: E402
    GUIDANCE_SCALE,
    IMAGES_PER_PROMPT,
    MODEL_ID,
    NUM_CLASS_IMAGES,
    NUM_INFERENCE_STEPS,
    RESOLUTION,
    SUBJECTS,
    class_names,
    class_prompt,
    prompt_rows,
    sample_seed,
    subject_by_name,
)

PHASE1_DIR = ROOT / "phase1" / "outputs"
CLASS_DIR = ROOT / "phase2" / "class_images"


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline():
    chosen = device()
    # fp16 on the GPU. diffusers 0.39 takes torch_dtype.
    dtype = torch.float16 if chosen in {"mps", "cuda"} else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(chosen)
    pipe.set_progress_bar_config(disable=True)
    return pipe, chosen


def write_metadata(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "metadata.json"
    existing = []
    if path.exists():
        existing = json.loads(path.read_text())
    by_file = {row["file"]: row for row in existing}
    for row in rows:
        by_file[row["file"]] = row
    path.write_text(json.dumps(list(by_file.values()), indent=2))


def generate_one(pipe, prompt, seed, path):
    if path.exists() and path.stat().st_size > 0:
        print(f"skip {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=RESOLUTION,
        width=RESOLUTION,
        generator=generator,
    )
    result.images[0].save(path)
    print(f"wrote {path.relative_to(ROOT)}  seed={seed}  {prompt}")


def phase1(pipe, subjects, prompt_indices, samples=IMAGES_PER_PROMPT):
    for subject in subjects:
        rows = prompt_rows(subject)
        if prompt_indices is not None:
            rows = [row for row in rows if row["index"] in prompt_indices]
        folder = PHASE1_DIR / subject["name"]
        metadata = []
        for row in rows:
            for sample_index in range(samples):
                seed = sample_seed("phase1", subject["name"], row["index"], sample_index)
                filename = f"p{row['index']:02d}_s{sample_index}.png"
                path = folder / filename
                generate_one(pipe, row["phase1"], seed, path)
                metadata.append(
                    {
                        "file": filename,
                        "prompt_index": row["index"],
                        "sample_index": sample_index,
                        "prompt": row["phase1"],
                        "seed": seed,
                    }
                )
        write_metadata(folder, metadata)


def class_images(pipe, names):
    for class_name in names:
        prompt = class_prompt(class_name)
        folder = CLASS_DIR / class_name
        metadata = []
        for sample_index in range(NUM_CLASS_IMAGES):
            seed = sample_seed("class", class_name, 0, sample_index)
            filename = f"{sample_index:04d}.png"
            path = folder / filename
            generate_one(pipe, prompt, seed, path)
            metadata.append(
                {
                    "file": filename,
                    "prompt": prompt,
                    "sample_index": sample_index,
                    "seed": seed,
                }
            )
        write_metadata(folder, metadata)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--class-images", action="store_true")
    parser.add_argument("--subject", action="append", dest="subjects")
    parser.add_argument("--class-name", action="append", dest="class_names")
    parser.add_argument("--prompt-index", type=int, action="append", dest="prompt_indices")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One image: a dog on the beach.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.phase1 or args.class_images or args.smoke):
        raise SystemExit("Choose --smoke, --phase1, and/or --class-images.")

    pipe, chosen = load_pipeline()
    print(f"device={chosen} model={MODEL_ID}")

    if args.smoke:
        phase1(pipe, [subject_by_name("dog")], [2], samples=1)
        return

    if args.phase1:
        if args.subjects:
            subjects = [subject_by_name(name) for name in args.subjects]
        else:
            subjects = SUBJECTS
        phase1(pipe, subjects, set(args.prompt_indices) if args.prompt_indices else None)

    if args.class_images:
        names = args.class_names if args.class_names else class_names()
        class_images(pipe, names)


if __name__ == "__main__":
    main()
