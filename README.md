# OASIS-RC v2 crack-segmentation research package

This branch contains the source-anchored reconstructed OASIS-RC v2 implementation for crack segmentation. The exact historical v2 snapshot was not preserved, so historical numbers must not be described as bit-for-bit reproducible from this tree.

```text
experiment_id          = oasis-rc-v2-relational-hard-negative
checkpoint_schema      = 3
method_version         = OASIS-RC-v2
implementation_version = 2.0.4
```

## Scientific contract

Official controlled arms:

```text
S0_control
  L = L_seg

S1_oasis_rc_v2
  L = L_seg + lambda_oasis * rc_ramp * L_RCv2

S2_aosk
  L = L_seg + lambda_aosk * L_AOSK_oriented

S3_oasis_rc_v2_aosk
  L = L_seg
    + lambda_oasis * rc_ramp * L_RCv2
    + lambda_aosk * L_AOSK_oriented
```

`L_seg = BCE + Dice`.

The canonical AOSK variant in 2.0.4 is the original source-anchored `oriented-consistency-v1`. The centerline/clDice implementation remains available only as an optional ablation; it is not the official S2/S3 objective.

OASIS-RC and AOSK are training-only. Deployment is always:

```text
RGB -> student -> crack logits/mask
```

Student deployment checkpoints contain student state only and reject critic/generator/discriminator/AOSK training state.

## OASIS-RC v2 critic

Input:

```text
RGB image + soft/binary mask
```

The critic keeps the inherited source architecture:

```text
separate RGB/mask encoders
-> fi, fm, fi*fm, abs(fi-fm)
-> relational fusion
-> crack / mismatch / pair-validity outputs
```

The semantic logits are the inherited hierarchical composition from crack and mismatch logits. Implementation 2.0.4 does not redesign the critic head structure.

## C1-C9 hard-negative contract

```text
C1 translation
C2 erosion
C3 dilation
C4 local crack break
C5 wrong width
C6 wrong connection / bridge
C7 non-self crack donor mask
C8 crack mask on true-normal RGB
C9 texture-guided false-positive blob
```

Hardening retained in 2.0.4:

- operator identity is preserved; no arbitrary pixel rescue;
- C7 requires a non-self crack-positive donor;
- C8 is valid only on explicit true-normal RGB with a crack-positive donor;
- C9 uses RGB texture when available;
- metadata records the actual returned operator;
- `torch.roll` is not used.

## Critic objective

```text
L_critic =
    L_semantic_balanced
  + lambda_crack    * L_valid_crack_dice
  + lambda_mismatch * L_mismatch
  + lambda_pair     * L_pair
  + lambda_rgb_mask * L_RGB_spatial_shuffle_pair
  + optional normal-RGB donor supervision
```

RGB spatial shuffle is pair-validity-only. Mask flip is a qualification diagnostic (`mask_pair_drop`) and is not part of critic optimizer gradients.

Critic checkpoints bind:

```text
rgb_shuffle_pair_only = true
mask_flip_training     = false
mask_variant_contract  = operator-preserved-v1
```

The validator is fail-closed: missing contract metadata is rejected rather than synthesized.

## Student RC objective

The reconstructed student objective remains unchanged:

```text
rank(pred, GT)
+ rank(pred, corrupted)
+ background false-positive penalty
```

GT/corrupted energies are detached and the critic is frozen during student optimization.

## Data integrity and test firewall

Retained protocol hardening:

```text
canonical manifest
-> split/lineage normalization
-> leakage + exact-duplicate repair
-> fail-closed CleanEval
-> dataset-byte-bound Gate 0
-> certified train/val view
-> validation-only model/threshold selection
-> final canonical test exactly once after protocol lock
```

Official training code refuses canonical test rows.

## N0 and N25 are explicit protocols

### N0 — crack-only primary protocol

N0 does **not** require `NORMAL_ROOT`.

```bash
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export DATA_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_n0/data
bash scripts/prepare_n0_data.sh
```

### N25 — true-normal RGB extension

N25 requires an audited true-normal source:

```bash
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
export DATA_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_n25/data
bash scripts/prepare_real_data.sh
```

Do not describe N25 S0 as the historical crack-only control. N0 and N25 are separate experiments.

## One-seed acceptance

Start with seed 1337.

### N0

```bash
export EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_n0/seed_1337
export DATA_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_n0/data
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export SEED=1337
export NORMAL_FRACTION=0.0
export CRITIC_EPOCHS=10
export EPOCHS=12
export LAMBDA_OASIS=0.001
export LAMBDA_AOSK=0.01
export DETERMINISM_MODE=best_effort
bash scripts/run_training_ready.sh 2>&1 | tee "$EXP_ROOT/training_ready.log"
```

### N25

Use the same command with:

```bash
export NORMAL_FRACTION=0.25
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
```

Success ends with:

```text
TRAINING_PIPELINE_READY seed=1337
AOSK_VARIANT=oriented-consistency-v1
TEST_FIREWALL=CLOSED
```

## Critic qualification

Base gates include:

```text
valid_crack_recall            >= 0.80
invalid_recall                >= 0.90
rgb_pair_drop                 >= 0.05
mask_pair_drop                >= 0.05
min_corruption_invalid_recall >= 0.70
no background-only collapse
```

N25 additionally requires true-normal/C8 qualification evidence. Normal-domain diagnostics from `normal_train` are training-domain qualification metrics and must not be described as independent held-out generalization.

## Full smoke without opening canonical test

Example N0 smoke:

```bash
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/full_smoke_n0_1337
export NORMAL_FRACTION=0.0
bash scripts/run_full_smoke_train_test.sh
```

For N25 add `NORMAL_ROOT` and set `NORMAL_FRACTION=0.25`.

Success ends with:

```text
FULL_SMOKE_PASS
AOSK_VARIANT=oriented-consistency-v1
SMOKE_TEST_SPLIT=smoke_test
CANONICAL_TEST_OPENED=NO
TEST_FIREWALL=CLOSED
```

`smoke_test` is derived from train/validation-domain rows and is not the canonical benchmark test.

## Three seeds

Canonical seeds are exactly:

```text
1337
2027
31415
```

Use only `scripts/run_all_seeds.sh`. The obsolete `run_three_seeds.sh` runner has been removed.

### N0

```bash
export BASE_EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_3seed_n0
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export NORMAL_FRACTION=0.0
bash scripts/run_all_seeds.sh
```

### N25

```bash
export BASE_EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_3seed_n25
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export NORMAL_FRACTION=0.25
bash scripts/run_all_seeds.sh
```

## CUDA reproducibility

Canonical GPU runners use:

```text
DETERMINISM_MODE=best_effort
CUBLAS_WORKSPACE_CONFIG=:4096:8
cuDNN benchmark=false
cuDNN deterministic=true
TF32=false
```

Run/checkpoint metadata records the relevant runtime and data provenance.

## Final test exactly once

Training, smoke and validation keep the canonical test closed. After architecture, checkpoint, hyperparameters and threshold are frozen, create the protocol lock and run:

```bash
bash scripts/run_final_test.sh /absolute/path/PROTOCOL_LOCK.json
```

The final-test runner atomically marks the test opened before reading canonical test data and has no normal paper replay path.

## CI versus host acceptance

GitHub CI checks installation, compileall, pytest, canonical method-contract assertions and shell syntax. It does not certify the real `/hdd1/...` data or NVIDIA A30 runtime.

Required acceptance sequence:

```text
GitHub CI PASS
-> real-data Gate 0 PASS
-> A30/CUDA preflight PASS
-> critic qualification PASS
-> seed-1337 S0-S3 smoke/validation PASS
-> N0 and/or N25 three-seed validation
-> freeze protocol
-> canonical test once
```

Never lower a data or critic gate after observing experiment outcomes.
