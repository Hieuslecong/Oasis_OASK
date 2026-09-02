# Final validation report â current workspace

## Correctness

| Gate | Status | Evidence |
|---|---|---|
| Existing + new unit tests | PASS | 35/35 CPU tests |
| Static compile | PASS | `compileall` |
| Toy leakage checks | PASS | 24-row split-specific toy manifest |
| Real leakage check | BLOCKED | manifest paths are unavailable |
| Smoke training | PASS | toy baseline and hard-constrained forward/backward |
| Overfit test | PASS | 2 samples: loss 1.6403 -> 0.00044 |
| Checkpoint/resume | PASS | resumed epoch 3 from saved epoch 2 |
| Real-data training | BLOCKED | no real images/masks |

## Generator

- DP-GAN semantic preservation: **INCONCLUSIVE**; no checkpoint/backend.
- DP-GAN factor control: **INCONCLUSIVE**; G1 not run.
- DP-GAN realism: **INCONCLUSIVE**; no real renderer output.
- Toy renderer geometry/control: **PASS as integration-only evidence**.

## Hypotheses

| Hypothesis | Status |
|---|---|
| H1 real calibration improves robustness | INCONCLUSIVE |
| H2 factorized nuisance control is valid | INCONCLUSIVE |
| H3 hard search finds useful valid counterfactuals | INCONCLUSIVE |
| H4 geometry envelope prevents label corruption | INCONCLUSIVE on DP-GAN; PASS on toy integration |
| H5 cross-domain robustness improves | INCONCLUSIVE |
| H6 gains generalize across backbones | INCONCLUSIVE |

## Stop conditions

No final paper metric or success criterion is claimed. Real-data Gate 0,
DP-GAN qualification, multi-seed experiments, cross-domain evaluation, and
the G0/G1/G2 ablation remain mandatory before scientific conclusions.
