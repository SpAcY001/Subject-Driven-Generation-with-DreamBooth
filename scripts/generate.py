"""Phase 1 images and class images from pretrained Stable Diffusion, and Phase 2 images
from the fine-tuned DreamBooth models."""

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
PHASE2_MODEL_DIR = ROOT / "phase2" / "outputs"
PHASE2_DIR = ROOT / "phase2" / "samples"


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(model=MODEL_ID):
    chosen = device()
    # fp16 on the GPU. diffusers 0.39 takes torch_dtype.
    dtype = torch.float16 if chosen in {"mps", "cuda"} else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype)
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


def phase2(subjects, prompt_indices, samples=IMAGES_PER_PROMPT):
    for subject in subjects:
        model = PHASE2_MODEL_DIR / subject["name"]
        if not (model / "model_index.json").is_file():
            raise SystemExit(f"no trained model in {model.relative_to(ROOT)}. Run phase2/train.sh first.")
        pipe, _ = load_pipeline(model)
        rows = prompt_rows(subject)
        if prompt_indices is not None:
            rows = [row for row in rows if row["index"] in prompt_indices]
        folder = PHASE2_DIR / subject["name"]
        metadata = []
        for row in rows:
            for sample_index in range(samples):
                # Same seed as the Phase 1 image, so each pair starts from the same noise
                # and differs only by the fine-tune.
                seed = sample_seed("phase1", subject["name"], row["index"], sample_index)
                filename = f"p{row['index']:02d}_s{sample_index}.png"
                path = folder / filename
                generate_one(pipe, row["phase2"], seed, path)
                metadata.append(
                    {
                        "file": filename,
                        "prompt_index": row["index"],
                        "sample_index": sample_index,
                        "prompt": row["phase2"],
                        "phase1_prompt": row["phase1"],
                        "seed": seed,
                    }
                )
        write_metadata(folder, metadata)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    parser.add_argument("--phase2", action="store_true", help="Needs the models from phase2/train.sh.")
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
    if not (args.phase1 or args.class_images or args.phase2 or args.smoke):
        raise SystemExit("Choose --smoke, --phase1, --class-images, and/or --phase2.")

    if args.subjects:
        subjects = [subject_by_name(name) for name in args.subjects]
    else:
        subjects = SUBJECTS
    prompt_indices = set(args.prompt_indices) if args.prompt_indices else None

    if args.phase2:
        # Each subject has its own fine-tuned model, loaded inside phase2().
        print(f"device={device()} models={PHASE2_MODEL_DIR.relative_to(ROOT)}")
        phase2(subjects, prompt_indices)
        if not (args.phase1 or args.class_images or args.smoke):
            return

    pipe, chosen = load_pipeline()
    print(f"device={chosen} model={MODEL_ID}")

    if args.smoke:
        phase1(pipe, [subject_by_name("dog")], [2], samples=1)
        return

    if args.phase1:
        phase1(pipe, subjects, prompt_indices)

    if args.class_images:
        names = args.class_names if args.class_names else class_names()
        class_images(pipe, names)


if __name__ == "__main__":
    main()
