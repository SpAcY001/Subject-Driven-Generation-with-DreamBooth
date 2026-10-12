# Subject-Driven Generation with DreamBooth

CMPE 261 project. Fine-tune Stable Diffusion so one specific subject stays the same when the text prompt changes.

A normal prompt such as "a dog on the beach" draws a new dog every time. DreamBooth takes a few photos of one subject, binds them to a unique token, and can then place that same subject in new scenes.

## Team

- Ritwik Reddy Konuganti (019151535)
- Naga Vijay Pavan Singampalli (019151743)

## Plan

- Backbone: Stable Diffusion v1.5, through Hugging Face Diffusers
- Phase 1: pretrained Stable Diffusion, no fine-tuning
- Phase 2: DreamBooth with the prior-preservation loss
- Phase 3: DreamBooth with LoRA, plus a prior-preservation ablation
- Data: the [DreamBooth dataset](https://github.com/google/dreambooth) (30 subjects). Evaluation uses 8 of them.
- Metrics: DINO and CLIP-I for subject fidelity, CLIP-T for prompt fidelity

## Papers

- Ruiz et al. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023. https://arxiv.org/abs/2208.12242
- Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
