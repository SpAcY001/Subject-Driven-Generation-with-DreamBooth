# Subject-Driven Generation with DreamBooth

CMPE 261 project. We fine-tune Stable Diffusion so one subject stays the same when the prompt changes.

"a dog on the beach" gives a new dog every time. We use a few photos of one subject and a token, `zwx`. After training, "a zwx dog on the beach" should be that same dog.

## Team

- Ritwik Reddy Konuganti (019151535)
- Naga Vijay Pavan Singampalli (019151743)

## Where we are

Phase 1 and Phase 2 are done. LoRA has not started.

- 8 subjects, 43 photos in `data/references/`. The 512 crops used for training are in `data/references_512/`.
- 800 pretrained images: 8 subjects x 25 prompts x 4. Prompts are class-only (`a dog on the beach`, no `zwx`). They live in `phase1/outputs/{subject}/`.
- 1,200 class images, 200 each for dog, cat, backpack, clock, teapot, and sneaker, in `phase2/class_images/`. dog2 reuses dog, cat2 reuses cat. These are the prior-preservation inputs. We do not score them.
- One DreamBooth model per subject in `phase2/outputs/{subject}/`. Prior preservation on. Text encoder trained. 400 steps, learning rate 5e-6, prior loss weight 1.0, 200 class images, resolution 512.
- Another 800 images from those models, prompts with the token kept (`a zwx dog on the beach`), in `phase2/outputs/{subject}/images/`.
- Scores are in `phase1/metrics.json` and `phase2/metrics.json`.

The pngs and the model weights stay on the HPC. They are gitignored.

The notebook shows the photos, the Phase 1 dog figure, the same four prompts after the dog fine-tune, and a beach row for the other seven subjects. The full 800 + 800 are in the output folders.

We did not keep the first dog settings. UNet only, 800 steps, 5e-6, put the dog on the beach but it was often a different dog. Training the text encoder for 800 steps kept the corgi and then copied the training photos. 400 steps at 5e-6 with the text encoder on is the run that held the corgi and still changed the scene. All eight subjects use that.

## Scores

DINO and CLIP-I compare each generated image with the real photos of that same subject, then we average. CLIP-T compares the image with the prompt stored in that image's `metadata.json`. DINO is `facebook/dino-vits16` (CLS token). CLIP is `openai/clip-vit-base-patch32`. Phase 2 prompts include `zwx`.

| | DINO | CLIP-I | CLIP-T |
|---|---:|---:|---:|
| Phase 1, all 800 | 0.351 | 0.732 | 0.329 |
| Phase 2, all 800 | 0.695 | 0.874 | 0.264 |
| Paper, fine-tuned SD | 0.668 | 0.803 | 0.305 |

Phase 1, per subject:

| | DINO | CLIP-I | CLIP-T |
|---|---:|---:|---:|
| dog | 0.205 | 0.694 | 0.330 |
| dog2 | 0.115 | 0.703 | 0.332 |
| cat | 0.516 | 0.759 | 0.334 |
| cat2 | 0.506 | 0.740 | 0.328 |
| backpack | 0.334 | 0.777 | 0.327 |
| clock | 0.335 | 0.746 | 0.314 |
| teapot | 0.383 | 0.772 | 0.355 |
| colorful_sneaker | 0.416 | 0.664 | 0.311 |

Phase 2, per subject:

| | DINO | CLIP-I | CLIP-T |
|---|---:|---:|---:|
| dog | 0.825 | 0.882 | 0.276 |
| dog2 | 0.800 | 0.886 | 0.245 |
| cat | 0.847 | 0.929 | 0.247 |
| cat2 | 0.790 | 0.890 | 0.258 |
| backpack | 0.572 | 0.868 | 0.283 |
| clock | 0.534 | 0.853 | 0.257 |
| teapot | 0.408 | 0.842 | 0.256 |
| colorful_sneaker | 0.786 | 0.839 | 0.291 |

DINO is the identity number. Phase 1 draws a dog, but not our dog, so DINO stays low. Phase 2 is above the paper's fine-tuned Stable Diffusion on DINO and CLIP-I. CLIP-T dropped, which is the usual tradeoff once the token sticks to the subject. The teapot is the weak one. The cats and the first dog are the strong ones.

## Repo

- `data/references/` original photos
- `data/references_512/` training crops
- `configs/subjects.py` subjects, token `zwx`, and the 25 prompts
- `notebooks/01_explore_data.ipynb` photos, the dog figure, and the beach comparison
- `scripts/generate.py` Phase 1, class images, and Phase 2
- `scripts/score_phase1.py` and `scripts/score_phase2.py`
- `phase1/metrics.json` pretrained means
- `phase2/metrics.json` DreamBooth means
- `phase2/train.sh` the training command we actually ran
- `phase2/train_dreambooth.py` Diffusers 0.39 example, with a small skip so `metadata.json` is not opened as a photo

Live subjects use the outfit prompt list. Objects use the scene prompt list. One token for every subject, because each subject is its own fine-tune.

## HPC

Jobs run on a GPU node, not the login node. The venv is `.venv` and it picks up the cluster PyTorch (`ml/torch/2.6`). GPU nodes have no internet, so model weights get downloaded on `coe-hpc1` first. We skip `cs002` and `cs001`. Those A100s refuse CUDA calls. Training ran on an A100. The H100s were free for the Phase 2 images, so that job used one.

```bash
sbatch scripts/generate.sbatch
sbatch scripts/score_phase1.sbatch
SUBJECT=dog sbatch phase2/train.sbatch
sbatch scripts/generate_phase2.sbatch
sbatch scripts/score_phase2.sbatch
```

Generation skips files that are already there. Do not rerun it unless some pngs are missing.

## Next

1. DreamBooth-LoRA, prior preservation on, same 8 subjects, same prompts, then score another 800. This is the cheap version of Phase 2.
2. Ablation on two subjects only, dog and backpack. LoRA again, prior preservation off. Same photos and prompts, no class images. Those pictures should stick to the original pose and background. Compare them with the LoRA runs that had the safeguard on.
3. A 6–8 frame storyboard of the same dog in new scenes. That one uses the trained model. It is the demo, not another score column.

## Papers

- Ruiz et al. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023. https://arxiv.org/abs/2208.12242
- Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
