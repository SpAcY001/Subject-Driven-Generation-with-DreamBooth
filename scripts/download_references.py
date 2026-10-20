"""Download the eight DreamBooth subject folders used in this project."""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.subjects import SUBJECTS  # noqa: E402

API = "https://api.github.com/repos/google/dreambooth/contents/dataset"
RAW = "https://raw.githubusercontent.com/google/dreambooth/main/dataset"
DEST = ROOT / "data" / "references"


def list_images(subject_name):
    with urllib.request.urlopen(f"{API}/{subject_name}") as response:
        entries = json.load(response)
    return [entry["name"] for entry in entries if entry["type"] == "file"]


def download():
    DEST.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        name = subject["name"]
        folder = DEST / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename in list_images(name):
            target = folder / filename
            if target.exists() and target.stat().st_size > 0:
                print(f"skip {name}/{filename}")
                continue
            url = f"{RAW}/{name}/{filename}"
            print(f"get  {name}/{filename}")
            urllib.request.urlretrieve(url, target)
    print(f"saved under {DEST}")


if __name__ == "__main__":
    download()
