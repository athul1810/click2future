# Research references

These are starting references for implementation and the report. They do not establish that the proposed combination is new. T18 should also compare closely related action-conditioned video prediction and physics-supervised generation methods before making a novelty claim.

## Models

| Reference | Use in this project |
|---|---|
| Ho, Jain, and Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020 | Denoising objective and sampling background for E1-E4 |
| Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022 | Autoencoder and latent-space diffusion background |
| Ho et al., [Video Diffusion Models](https://arxiv.org/abs/2204.03458), 2022 | Video diffusion and temporal model background |
| Sohn, Lee, and Yan, [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html), NeurIPS 2015 | Conditional VAE baseline E0 |

Our proposed compact model is an adaptation for synthetic tabletop clips. The architecture and training budget still need to be tested on the available hardware.

## Metrics

| Reference | Use in this project |
|---|---|
| Wang et al., [Image Quality Assessment: From Error Visibility to Structural Similarity](https://www.cns.nyu.edu/~lcv/ssim/), 2004 | SSIM; the linked author page includes the paper and implementations |
| Zhang et al., [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924), CVPR 2018 | LPIPS |
| Unterthiner et al., [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717), 2018 preprint | FVD, if included as a secondary metric |

Trajectory errors, collision-event scoring, and action controls use the definitions in [EVALUATION.md](EVALUATION.md). Record the implementation version and preprocessing for each metric rather than assume all implementations are comparable.

## Dataset context

[BAIR robot-pushing, TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small) documents a related action/video dataset. Our planned training data comes from the tabletop simulator; BAIR is background context unless the team explicitly changes that decision.

## Notes for the report

The [14 October primary-source review](SOTA_2026-10-14.md) adds PhyCo, PhysCtrl,
PlaySlot, DIAMOND, Diffusion Forcing, recent causal/reactivity benchmarks, and the
representative foundation-model frontier. Code revisions and paper versions are in
[research_sources.json](research_sources.json). Availability is verified, but those
external systems have not been independently installed/trained/evaluated by this project.

- Keep these source links in the working repository. Use bibliographic entries without external URLs or clickable external links in the submitted PDFs.
- Cite the method actually implemented and identify any borrowed code or pretrained weights.
- Compare each related method's inputs, output representation, physical supervision, and uncertainty treatment with Click2Future.
- Describe object-centric conditioning and auxiliary supervision as proposed contributions until the literature review and experiments support a stronger claim.
- Keep the chosen model size, dataset size, and compute budget separate from those used in the reference papers.
