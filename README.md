# Subject-Driven Generation with DreamBooth

CMPE 261 project. We fine-tune Stable Diffusion so one subject stays the same when the prompt changes.

"a dog on the beach" gives a new dog every time. We use a few photos of one subject and a token, `zwx`. After training, "a zwx dog on the beach" should be that same dog.

## Team

- Ritwik Reddy Konuganti (019151535)
- Naga Vijay Pavan Singampalli (019151743)

## Repo

- `data/references/` original photos for the 8 subjects
- `data/references_512/` same photos, short side resized to 512, then center-cropped
- `configs/subjects.py` subject list, token, and the 25 prompts
- `notebooks/01_explore_data.ipynb` the photos, the prompts, and a few pretrained dog images
- `scripts/generate.py` Phase 1 images and class images
- `phase2/train.sh` training command, not run yet

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_references.py
python3 scripts/preprocess_references.py
```

## Phase 1

Phase 1 prompts use the class name only, like `a dog on the beach`. No token.

Full run is 8 subjects x 25 prompts x 4 images (800). Class images are 200 each for dog, cat, backpack, clock, teapot, and sneaker. The notebook only has four dog prompts so far: beach, snow, red hat, jungle.

```bash
python3 scripts/generate.py --phase1
python3 scripts/generate.py --class-images
```

Images that are already saved get skipped.

## Phase 2

One subject at a time, on the HPC. Training reads `data/references_512/`. Class images go in `phase2/class_images/`.

```bash
SUBJECT=dog bash phase2/train.sh
```

Needs `train_dreambooth.py` from the Diffusers examples.

## Papers

- Ruiz et al. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023. https://arxiv.org/abs/2208.12242
- Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
