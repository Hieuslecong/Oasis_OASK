# OASIS-RC-v2.1 crack-segmentation research package

This branch contains the **OASIS-RC-v2.1 development revision**. The reconstructed v2.0.4 code remains historical engineering evidence only; it is not the source of truth for this branch.

```text
experiment_id          = oasis-rc-v2.1-gt-anchored-relational-energy-head
checkpoint_schema      = 5
method_version         = OASIS-RC-v2.1
implementation_version = 2.1.0-dev2
```

Scientific source of truth: `METHOD_SPEC_V2_1.md`  
Executable protocol: `protocols/real_data_v21.json`

## Scientific question

Can training-only relational and structure-aware supervision improve accuracy, topology-sensitive outcomes and false-positive robustness of lightweight RGB-only crack segmentation without increasing inference complexity?

Deployment is always:

```text
RGB -> lightweight student -> crack logits/mask
```

The critic, relation-energy head and AOSK are training-only and are forbidden in deployment checkpoints.

## Canonical arms

```text
B0  BCE + Dice
B1  BCE + Dice + clDice
B2  BCE + Dice + frozen pretrained pair-critic BCE
S1  BCE + Dice + OASIS-RC-v2.1
S2  BCE + Dice + AOSK structure-tensor-v2
S3  BCE + Dice + OASIS-RC-v2.1 + AOSK structure-tensor-v2
```

B2 is an ablation using a frozen pretrained pair critic. It is **not** conventional jointly-trained adversarial training and must not be described as such.

Primary paired contrasts:

```text
B1 - B0
B2 - B0
S1 - B0
S2 - B0
S3 - S2
```

Negative/null effects are valid outcomes and must not be rescued by post-hoc tuning.

## OASIS-RC-v2.1 relation energy

The critic exposes a dedicated scalar energy head. Lower energy means a more compatible RGB-mask relation.

```text
E_G = E(I, GT)
E_P = E(I, student prediction)
E_C = E(I, structured corruption)
```

Critic calibration includes:

```text
GT energy anchor
GT < corruption endpoint ordering
continuous GT -> corruption path ordering
```

Student relational supervision uses a GT anchor plus one-sided corruption rejection. GT/corruption energies are detached and the critic is frozen during student optimization.

A connected arm is allowed only after both representation/classification qualification and relation-energy qualification pass. Every v2.1 critic-consuming launch re-measures qualification from the currently loaded weights; a stored PASS flag is provenance, not sole authorization.

## AOSK dev2

Canonical AOSK is `structure-tensor-steered-v2`.

It uses the full 2x2 structure tensor (`Jxx`, `Jyy`, `Jxy`), derives the local tangent orientation and samples logits at `+/- tangent` using differentiable bilinear sampling. Low-coherence areas smoothly fall back to isotropic local consistency.

AOSK is **not a topology loss**. Topology, continuity or junction claims require independent topology metrics and comparison against B1/clDice.

## Data protocols

### N0

No external normal supervision. Certified native-empty rows from the crack dataset may remain internal true negatives.

### N25

External true-normal RGB is used at 25% batch composition. Normal data must be lineage-disjoint across:

```text
normal_train
normal_val
normal_test
```

N25 critic qualification requires held-out `normal_val`; it may not silently fall back to `normal_train`.

The v2.1 preparation path is:

```text
canonical manifest
-> clean_manifest.py
-> CleanEval
-> audited external normal source
-> lineage-safe normal split
-> full-benchmark Gate0
-> N0 / N25 training views
-> training-view Gate0 bound to the full certificate
```

Use:

```bash
export DATA_ROOT=/path/to/data_v21
export CANONICAL_MANIFEST=/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/path/to/true_normal_rgb
export LINEAGE_REGEX='...'
export PYTHON=/path/to/python
bash scripts/prepare_real_data_v21.sh
```

The unified preparation script creates both N0 and N25 views. `test` and `normal_test` never enter either training view.

## Development training

Canonical order:

```text
Gate0
-> CUDA preflight
-> shared student initialization
-> S0/B0 baseline
-> critic training
-> critic representation + energy qualification
-> trained-S0 RC gradient/energy diagnostic
-> auxiliary-weight development calibration/freeze
-> B0/B1/B2/S1/S2/S3
-> crack-val and normal-val evaluation separately
-> development GO/KILL decision
```

One-command development launcher:

```bash
export NORMAL_FRACTION=0.0   # or 0.25
export SEED=1337
export EXP_ROOT=/path/to/experiments/v21/seed1337
export DATA_ROOT=/path/to/data_v21
export CANONICAL_MANIFEST=/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/path/to/true_normal_rgb
export LINEAGE_REGEX='...'
export PYTHON=/path/to/python
bash scripts/run_training_ready_v21.sh
```

`run_training_ready_v21.sh` is development-only. Canonical final-test access remains closed.

## Critic development gates

At minimum:

```text
valid_crack_recall                 >= 0.80
invalid_recall                     >= 0.90
rgb_pair_drop                      >= 0.05
mask_pair_drop                     >= 0.05
min_corruption_invalid_recall      >= 0.70
samples_per_required_corruption    >= 16
positive_energy_gap_fraction       >= 0.70
continuous_path_order_fraction     >= 0.65
mean_energy_gap                    > 0
true global median_energy_gap      > 0
energy_samples                     >= 16
energy_finite                      = true
```

Thresholds are development gates and must be frozen before confirmatory runs; never lower them to obtain a PASS.

## Auxiliary-weight discipline

Parser lambdas are development starting points, not confirmatory hyperparameters. dev2 logs both:

```text
||grad L_aux|| / ||grad L_seg||
lambda * ramp * ||grad L_aux|| / ||grad L_seg||
cosine(grad L_seg, grad L_aux)
```

Use development seed 1337 to calibrate auxiliary coefficients under one declared rule, then freeze them. Heterogeneous losses are not required to have identical gradient norms; the telemetry is a strength/stability diagnostic.

## Evaluation

Crack-overlap and topology metrics are computed on crack-positive rows only:

```text
precision / recall / Dice / IoU
clDice / skeleton precision / skeleton recall
component excess
```

True-negative rows are evaluated separately:

```text
normal FP pixels
normal FP components
normal any-FP rate
```

Do not average Dice/IoU structural zeros over true-negative rows.

## Statistics

Confirmatory uncertainty uses **training seed as the sampling unit**. `scripts/analyze_v21_paired.py` expects an index mapping each arm/seed to its crack and normal evaluation JSONs and reports:

```text
paired seed deltas
mean / std / median
seed-bootstrap 95% CI
Cohen dz
direction consistency
exact paired sign-flip p-value (secondary evidence)
Holm correction within metric families
```

Current confirmatory seed set:

```text
2027
31415
42421
51511
62617
```

Seed 1337 is development evidence and is excluded after any tuning.

## Final-test firewall

Both of the following are canonical final-test material:

```text
test
normal_test
```

Development evaluators refuse either split. Canonical evaluation is permitted only through `scripts/run_final_bundle_v21.py` after all models, thresholds, metrics, protocol and statistical plan are frozen.

The final bundle requires all six arms for every seed. Its ID is content-addressed rather than path-derived. The final runner requires a stable external `--ledger-root` and evaluates `test` plus `normal_test` under the same single-open marker.

## CI and actual CLI smoke

`OASIS-RC-v2 CI` performs compile, behavioural pytest regression tests, all shell parsing and protocol/version identity checks.

`OASIS-RC-v2.1 Actual Smoke` creates actual PNG/mask files on disk and executes production CLIs for:

```text
N0:  critic + B0/B1/B2/S1/S2/S3 + crack validation
N25: critic + B0/B1/B2/S1/S2/S3 + crack validation + normal_val FP evaluation
```

This is mechanical integration evidence only. It does not establish real-data efficacy.

## Real-data readiness rule

Before a full development experiment is called GO, the target dataset/host must pass:

```text
real CleanEval + Gate0
N0/N25 training-view certificates
CUDA preflight under the same determinism mode as training
shared initialization
trained baseline
critic representation + energy qualification
trained-S0 gradient/energy diagnostic
frozen auxiliary-weight development decision
six-arm development run
separate crack/normal evaluation
```

Confirmatory experiments remain NO-GO until all of the above, multi-seed analysis, and the immutable final bundle are frozen and verified.
