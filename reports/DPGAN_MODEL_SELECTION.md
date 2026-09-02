# DP-GAN model selection

## Decision

Use the official `sj-li/DP_GAN` generator as the renderer candidate, not as a
segmentation backbone. Keep existing OASIS-RC/AOSK code as a separate
reference baseline. Qualify the generator before hard-search experiments.

## Required variants

| Variant | Meaning | Current status |
|---|---|---|
| G0 | original DP-GAN latent/noise behavior | not run: checkpoint absent |
| G1 | explicit nuisance conditioning | adapter contract exists; backend absent |
| G2 | G1 plus regularization | not implemented until G1 evidence exists |

DP-GAN's latent noise must not be labelled disentangled. A factor is retained
only if single-factor sweeps change the target background feature, preserve
crack geometry, and avoid excessive non-target drift. If this fails, DP-GAN
remains a G0 comparator and the framework does not claim factorization.
