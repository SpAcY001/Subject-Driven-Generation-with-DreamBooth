# Subject-Driven Generation with DreamBooth

CMPE 261 project. We fine-tune Stable Diffusion so one subject stays the same when the prompt changes.

"a dog on the beach" gives a new dog every time. We use a few photos of one subject and a token, `zwx`. After training, "a zwx dog on the beach" should be that same dog.

## Team

- Ritwik Reddy Konuganti (019151535)
- Naga Vijay Pavan Singampalli (019151743)

## Where we are

Check-in 2. Phase 1 is done. Training has not started.

- 8 subjects, 43 photos in `data/references/`. The 512 crops used for training are in `data/references_512/`.
- 800 pretrained images on the cluster: 8 subjects x 25 prompts x 4. Prompts are class-only (`a dog on the beach`, no `zwx`). They live in `phase1/outputs/{subject}/`.
- 1,200 class images, 200 each for dog, cat, backpack, clock, teapot, and sneaker, in `phase2/class_images/`. dog2 reuses dog, cat2 reuses cat. These are the prior-preservation inputs. We do not score them.
- Phase 1 scores are in `phase1/metrics.json`.

The pngs stay on the HPC. They are gitignored.

The notebook still only shows the photos and 16 pretrained dog images (beach, snow, red hat, jungle). The full 800 are in `phase1/outputs/`.

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

## Repo

- `data/references/` original photos
- `data/references_512/` training crops
- `configs/subjects.py` subjects, token `zwx`, and the 25 prompts
- `notebooks/01_explore_data.ipynb` photos and a few pretrained dog images
- `scripts/generate.py` Phase 1 images and class images
- `scripts/score_phase1.py` the three scores
- `phase1/metrics.json` pretrained means, overall and per subject
- `phase2/train.sh` training command, not run yet

Live subjects use the outfit prompt list. Objects use the scene prompt list. One token for every subject, because each subject is its own fine-tune.

## HPC

Jobs run on a GPU node, not the login node. The venv is `.venv` and it picks up the cluster PyTorch (`ml/torch/2.6`). GPU nodes have no internet, so model weights get downloaded on `coe-hpc1` first. We skip `cs002` because that A100 fails CUDA calls. Maintenance is Friday, October 23, 10:00–11:00 AM.

```bash
sbatch scripts/generate.sbatch
sbatch scripts/score_phase1.sbatch
```

Both ask for one A100. Generation skips files that are already there. Do not rerun generation unless some pngs are missing.

## Next

1. Full DreamBooth, prior preservation on, one subject at a time: `SUBJECT=dog bash phase2/train.sh`. This needs `train_dreambooth.py` from the Diffusers examples. The script already sets prior loss weight 1.0, lr 5e-6, about 800 steps, 200 class images, resolution 512.
2. Generate the Phase 2 images with the token kept in the prompt (`a zwx dog on the beach`). Another 800.
3. Score that set the same way, so the table has a DreamBooth column next to the pretrained one.
4. DreamBooth-LoRA, prior preservation on, same 8 subjects, then score it.
5. Train dog and backpack again with LoRA and prior preservation off.
6. A short storyboard of the dog across a few new scenes, once that fine-tune holds the subject.

## Papers

- Ruiz et al. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023. https://arxiv.org/abs/2208.12242
- Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
