# Main report outline

Output filename: `group_07_report.pdf`. Course: AI6132 Generative AI For Visual Synthesis. Project 1: Technical Review. Group ID: 7.

This is a drafting outline, not a completed report. Use at least four pages of substantive content and include only results the team has measured. The four main sections below follow page 8 of the course PDF.

## 1. Introduction

- Define action-conditioned video generation and the tabletop task.
- Explain why a first frame and force can have multiple futures when physics is hidden.
- State the review question and the selected methods: conditional VAE and video diffusion, with the planned conditioning/supervision variants.
- Summarize closely related work and identify what this comparison tests. Do not claim novelty from the project idea alone.

## 2. Methods

- Describe each compared method, its inputs, latent representation, objective, and sampling procedure.
- Explain the simulator, action definition, hidden properties, and ground-truth labels.
- Describe object-centric conditioning and auxiliary heads if implemented.
- Include a clear architecture/comparison diagram and describe shared controls across methods.

## 3. Experiments and analysis

### Dataset and setup

Document dataset size, scene-level splits, shape/texture/force ranges, training settings, sampling seeds, and TC2 hardware. Explain metric definitions and any tracking limitations.

### Method comparison

Compare at least two methods using our own experiments. Include the E0-E4 table if all planned variants are completed. State actual seed counts, training budgets, and omitted variants.

### Hyperparameter analysis

Report a small planned study, such as sampling-step count or auxiliary-loss weight. State which values were tested, what stayed fixed, and the quality/cost tradeoff. Choose settings using validation data, then report the fixed test protocol.

### Component analysis

Compare numeric versus spatial conditioning and spatial conditioning with versus without auxiliary supervision, where implemented. Separate component changes from differences in training budget or model size.

### Visualizations, strengths, and limitations

Embed sample-frame strips, action arrows, trajectory plots, collision examples, and representative failures directly in the PDF. Discuss action adherence, hidden-physics uncertainty, generalization, and compute limits. Compare diversity with physical plausibility.

## 4. Conclusion

- Answer the review question using the measured results.
- State what worked, where methods failed, and what remains uncertain.
- Discuss practical future extensions and possible improvements.

## References

Use numbered or author-year citations with complete bibliographic entries. Remove external URLs and hyperlink annotations from the submitted PDF. Source links in the working repository are for the team's research, not for inclusion in the final submission.
