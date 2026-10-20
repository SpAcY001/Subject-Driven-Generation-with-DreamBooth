"""Resize the shorter side to 512, then center-crop a 512x512 square."""

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.subjects import RESOLUTION, SUBJECTS  # noqa: E402

SOURCE = ROOT / "data" / "references"
DEST = ROOT / "data" / "references_512"


def center_crop_512(image):
    image = image.convert("RGB")
    width, height = image.size
    scale = RESOLUTION / min(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.LANCZOS,
    )
    width, height = resized.size
    left = (width - RESOLUTION) // 2
    top = (height - RESOLUTION) // 2
    return resized.crop((left, top, left + RESOLUTION, top + RESOLUTION))


def preprocess():
    count = 0
    for subject in SUBJECTS:
        source_dir = SOURCE / subject["name"]
        dest_dir = DEST / subject["name"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(
            path
            for path in source_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for path in paths:
            target = dest_dir / f"{path.stem}.png"
            cropped = center_crop_512(Image.open(path))
            cropped.save(target)
            count += 1
            print(f"{subject['name']}/{path.name} {Image.open(path).size} -> {cropped.size}")
    print(f"wrote {count} images to {DEST}")


if __name__ == "__main__":
    preprocess()
