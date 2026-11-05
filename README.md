# Subject-Driven Generation with DreamBooth

CMPE 261 project. We fine-tune Stable Diffusion so one subject stays the same when the prompt changes.

"a dog on the beach" gives a new dog every time. We use a few photos of one subject and a token, `zwx`. After training, "a zwx dog on the beach" should be that same dog.

## Team

- Ritwik Reddy Konuganti (019151535)
- Naga Vijay Pavan Singampalli (019151743)

## Where we are

Phase 1 and Phase 2 are done. Phase 3 (LoRA) is next.

- 8 subjects, 43 photos in `data/references/`. The 512 crops used for training are in `data/references_512/`.
- 800 pretrained images on the cluster: 8 subjects x 25 prompts x 4. Prompts are class-only (`a dog on the beach`, no `zwx`). They live in `phase1/outputs/{subject}/`.
- 1,200 class images, 200 each for dog, cat, backpack, clock, teapot, and sneaker, in `phase2/class_images/`. dog2 reuses dog, cat2 reuses cat. These are the prior-preservation inputs. We do not score them.
- Phase 1 scores are in `phase1/metrics.json`.
- 8 DreamBooth models, one per subject, in `phase2/outputs/{subject}/` (5.2 GB each).
- 800 Phase 2 images, same 25 prompts x 4 with `zwx` in the prompt (`a zwx dog on the beach`), in `phase2/samples/{subject}/`. Each one reuses the seed of the Phase 1 image with the same prompt and sample number.
- Phase 2 scores are in `phase2/metrics.json`.

The pngs and models stay on the HPC. They are gitignored.

`notebooks/01_explore_data.ipynb` shows the photos and 16 pretrained dog images. `notebooks/02_phase2_results.ipynb` shows the Phase 2 scores and the pretrained and fine-tuned images side by side.

## Phase 1 scores

Pretrained Stable Diffusion v1.5, no fine-tuning. DINO and CLIP-I compare each generated image with the real photos of that same subject, then we average. CLIP-T compares the image with the prompt stored in that image's `metadata.json`. DINO is `facebook/dino-vits16` (CLS token). CLIP is `openai/clip-vit-base-patch32`.

| | DINO | CLIP-I | CLIP-T |
|---|---:|---:|---:|
| All 800 | 0.351 | 0.732 | 0.329 |
| dog | 0.205 | 0.694 | 0.330 |
| dog2 | 0.115 | 0.703 | 0.332 |
| cat | 0.516 | 0.759 | 0.334 |
| cat2 | 0.506 | 0.740 | 0.328 |
| backpack | 0.334 | 0.777 | 0.327 |
| clock | 0.335 | 0.746 | 0.314 |
| teapot | 0.383 | 0.772 | 0.355 |
| colorful_sneaker | 0.416 | 0.664 | 0.311 |

DINO is the low number. That is the identity failure: the model draws a dog, but not our dog. The paper's fine-tuned Stable Diffusion scores, which we compare against later, are DINO 0.668, CLIP-I 0.803, CLIP-T 0.305.

## Phase 2 scores

Full DreamBooth, prior preservation on: 800 steps per subject on its photos with `a photo of zwx {class}`, plus 200 class images with `a photo of a {class}` at prior loss weight 1.0, lr 5e-6, resolution 512. Same scorer as Phase 1. For CLIP-T we drop `zwx` from the prompt, so both phases are scored against the same text.

| | DINO before | DINO after | CLIP-I before | CLIP-I after | CLIP-T before | CLIP-T after |
|---|---:|---:|---:|---:|---:|---:|
| All 800 | 0.351 | 0.648 | 0.732 | 0.830 | 0.329 | 0.306 |
| dog | 0.205 | 0.744 | 0.694 | 0.835 | 0.330 | 0.305 |
| dog2 | 0.115 | 0.660 | 0.703 | 0.819 | 0.332 | 0.309 |
| cat | 0.516 | 0.762 | 0.759 | 0.858 | 0.334 | 0.290 |
| cat2 | 0.506 | 0.751 | 0.740 | 0.842 | 0.328 | 0.299 |
| backpack | 0.334 | 0.541 | 0.777 | 0.841 | 0.327 | 0.316 |
| clock | 0.335 | 0.502 | 0.746 | 0.820 | 0.314 | 0.291 |
| teapot | 0.383 | 0.469 | 0.772 | 0.824 | 0.355 | 0.330 |
| colorful_sneaker | 0.416 | 0.752 | 0.664 | 0.805 | 0.311 | 0.305 |
| Paper, SD DreamBooth | | 0.668 | | 0.803 | | 0.305 |

- DINO almost doubles, and it goes up for every subject. The model now draws our dog, not a dog.
- The two dogs, the two cats and the sneaker end up at 0.66 to 0.76. The backpack, clock and teapot stay near 0.5. Their photos are taken from very different angles: with the same DINO, their real photos score only 0.601, 0.561 and 0.425 against each other, against 0.78 to 0.88 for the pets. The teapot's 0.469 is above that.
- CLIP-T drops from 0.329 to 0.306. That is the price of holding the subject: `a purple zwx dog` comes out as our dog in its own colors. 0.306 is the paper's level.
- Against the paper: DINO 0.648 vs 0.668, CLIP-I 0.830 vs 0.803, CLIP-T 0.306 vs 0.305. The paper averages 30 subjects, we have 8. Our real photos score 0.709 against each other on average, the paper's 0.774, so relative to the real photos we reach 91% and the paper 86%.

Training took about 6 minutes per subject on an A100 40GB in full precision. We regenerated the Phase 1 images with the same seeds before training, and their scores match `phase1/metrics.json` to three decimals.

## Repo

- `data/references/` original photos
- `data/references_512/` training crops
- `configs/subjects.py` subjects, token `zwx`, and the 25 prompts
- `notebooks/01_explore_data.ipynb` photos and a few pretrained dog images
- `notebooks/02_phase2_results.ipynb` Phase 2 scores, and pretrained vs fine-tuned images
- `scripts/generate.py` Phase 1 images, class images, and Phase 2 images (`--phase2`)
- `scripts/score_phase1.py` the three scores
- `scripts/score_phase2.py` the same scores for Phase 2
- `scripts/build_notebook.py`, `scripts/build_phase2_notebook.py` write the two notebooks
- `phase1/metrics.json` pretrained means, overall and per subject
- `phase2/train.sh` training command for one subject
- `phase2/train_dreambooth.py` the Diffusers v0.39.0 DreamBooth example, unchanged
- `phase2/train.sbatch` trains every subject as an array job, then makes its 100 images
- `phase2/metrics.json` DreamBooth means, overall and per subject

Live subjects use the outfit prompt list. Objects use the scene prompt list. One token for every subject, because each subject is its own fine-tune.

## HPC

Jobs run on a GPU node, not the login node. The venv is `.venv` and it picks up the cluster PyTorch (`ml/torch/2.6`). GPU nodes have no internet, so model weights get downloaded on `coe-hpc1` first. We skip `cs002` because that A100 fails CUDA calls. Maintenance is Friday, October 23, 10:00–11:00 AM.

```bash
sbatch scripts/generate.sbatch
sbatch scripts/score_phase1.sbatch
```

Both ask for one A100. Generation skips files that are already there. Do not rerun generation unless some pngs are missing.

Phase 2, after the class images exist:

```bash
sbatch phase2/train.sbatch
sbatch scripts/score_phase2.sbatch
```

`train.sbatch` is an array, one subject per task, and skips training when `phase2/outputs/{subject}/model_index.json` is already there. It also skips `cs001`: on November 4, four of its GPUs reported CUDA not available. Training needs the full `sd-legacy/stable-diffusion-v1-5` download, including `safety_checker`, and sets `CUDA_HOME` because the cluster PyTorch environment imports DeepSpeed. If home is short on space, point `HF_HOME` at scratch before `sbatch`.

## Next

1. DreamBooth-LoRA, prior preservation on, same 8 subjects, then score it.
2. Train dog and backpack again with LoRA and prior preservation off.
3. A short storyboard of the dog across a few new scenes. The Phase 2 dog model holds the subject, so it can be used now.

## Papers

- Ruiz et al. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023. https://arxiv.org/abs/2208.12242
- Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
