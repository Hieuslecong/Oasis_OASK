# OASIS-RC v2 crack-segmentation research package

This branch contains the strict reconstructed OASIS-RC v2 research implementation for crack segmentation. The exact historical v2 source snapshot was not preserved, so historical numbers must not be described as bit-for-bit reproducible from this tree.

```text
experiment_id          = oasis-rc-v2-relational-hard-negative
checkpoint_schema      = 2
method_version         = OASIS-RC-v2
implementation_version = 2.0.2
```

## Scientific contract

Official controlled arms:

```text
S0 Control
  L = L_seg

S1 OASIS-RC v2
  L = L_seg + lambda_oasis * rc_ramp * L_RCv2

S2 AOSK-Topology
  L = L_seg + lambda_aosk * L_centerline_clDice

S3 OASIS-RC v2 + AOSK-Topology
  L = L_seg
    + lambda_oasis * rc_ramp * L_RCv2
    + lambda_aosk * L_centerline_clDice
```

`L_seg = BCE + Dice`.

The official AOSK arm in implementation 2.0.2 is `centerline-cldice-v1`, a differentiable centerline/topology objective for thin crack networks. The previous `oriented-consistency-v1` implementation remains in the source tree only as an ablation/backward-comparison method; it is not the official S2/S3 method.

OASIS-RC and AOSK are training-only. Deployment is always:

```text
RGB -> student -> crack logits/mask
```

Student deployment checkpoints reject critic, generator, discriminator and AOSK training state.

## OASIS-RC v2 critic

Input:

```text
RGB image + soft/binary mask
```

The critic uses separate image/mask encoders with relational fusion:

```text
fi, fm, fi*fm, abs(fi-fm)
```

Outputs:

```text
semantic : valid background / valid crack / invalid relation
mismatch : pixel mismatch logit
pair     : pair-validity logit
```

The semantic logits are the inherited hierarchical OASIS-RC v2 formulation derived from crack and mismatch logits; implementation 2.0.2 does not redesign this inherited architecture.

## C1-C9 mask-variant contract

Online training variants:

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

Implementation 2.0.2 enforces **operator identity**:

- no arbitrary random-pixel rescue is allowed;
- a no-op operator is retried using that operator's own stochastic parameters;
- an unforced sample may resample another eligible operator only after failure;
- metadata records the actual operator that produced the returned mask;
- C7 requires a genuine non-self crack-positive donor;
- C8 is executed only on explicit true-normal RGB with an empty target and a crack-positive donor;
- an invalid forced-kind request is resampled to a legal operator and metadata records the resampling;
- C9 uses RGB texture when RGB is available;
- `torch.roll` is not used.

## Critic objective

```text
L_critic =
    L_semantic_balanced
  + lambda_crack    * L_valid_crack_dice
  + lambda_mismatch * L_mismatch
  + lambda_pair     * L_pair
  + lambda_rgb_mask * L_RGB-spatial-shuffle-pair
  + optional normal-RGB donor supervision
```

RGB spatial shuffle is a pair-validity negative. It gives no semantic/crack/mismatch gradient through that branch.

**Mask flip is qualification-only in implementation 2.0.2. It is not part of the critic optimizer objective.**

Critic checkpoints bind:

```text
rgb_shuffle_pair_only = true
mask_flip_training     = false
mask_variant_contract  = operator-preserved-v1
```

## Student RC objective

The reconstructed OASIS-RC v2 student objective is unchanged:

```text
rank(pred, GT)
+ rank(pred, corrupted)
+ background false-positive penalty
```

GT/corrupted reference energies are detached and the critic is frozen during student optimization.

## Data integrity and Gate 0

Canonical data path:

```text
canonical manifest
  -> split/lineage normalization
  -> cross-split leakage repair (test > val > train)
  -> exact same-split deduplication
  -> fail-closed CleanEval
  -> row-level N0 certification
  -> normal-RGB audit
  -> full Gate 0
  -> N0 and N25 train/val views
  -> training
```

Gate-0 certificates bind both manifest SHA256 and the exact referenced image/mask bytes through `dataset_content_sha256`.

Official trainers refuse manifests containing canonical `split=test`.

## Prepare real data

```bash
cd /hdd1/hieulc/Oasis_AOSK/OASIS-RC-v2
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export DATA_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_real/data
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
bash scripts/prepare_real_data.sh
```

Do not train unless this ends with `REAL_DATA_READY`.

## CUDA reproducibility

Canonical GPU runners default to:

```text
DETERMINISM_MODE=best_effort
CUBLAS_WORKSPACE_CONFIG=:4096:8
cuDNN benchmark=false
cuDNN deterministic=true
TF32=false
```

Run metadata records Python, PyTorch, CUDA, cuDNN, GPU, compute capability, Git SHA and determinism mode.

## One-seed real-data acceptance

Start with seed 1337. N0/N25 must be explicit.

```bash
export EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_real/seed_1337
export DATA_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_real/data
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export SEED=1337
export NORMAL_FRACTION=0.25   # N25; use 0.0 for N0
export CRITIC_EPOCHS=10
export EPOCHS=12
export LAMBDA_OASIS=0.001
export LAMBDA_AOSK=0.01
export DETERMINISM_MODE=best_effort
bash scripts/run_training_ready.sh 2>&1 | tee "$EXP_ROOT/training_ready.log"
```

## Critic qualification

Base gates include:

```text
valid_crack_recall            >= 0.80
invalid_recall                >= 0.90
rgb_pair_drop                 >= 0.05
mask_pair_drop                >= 0.05
min_corruption_invalid_recall >= 0.70
rgb_pair_samples              > 0
mask_pair_samples             > 0
no background-only collapse
```

For normal-supervised runs, additional gates include true-normal background recall, pair-valid confidence, bounded invalid rate and C8 coverage.

`mask_pair_drop` remains a no-gradient qualification diagnostic. The current normal-domain qualification uses `normal_train`; it must not be described as independent held-out normal generalization.

## Four official validation arms

```text
S0_control
S1_oasis_rc_v2
S2_aosk_topology
S3_oasis_rc_v2_aosk_topology
```

All arms share the same seed-specific student initialization; S1/S3 share one frozen qualified critic.

Validation-only evaluation:

```bash
export EXP_ROOT=/path/to/seed_1337
export MANIFEST=/path/to/trainval_manifest.jsonl
export ARM_ROOT="$EXP_ROOT/arms"
bash scripts/run_validation_eval.sh
```

## Full smoke without opening canonical test

```bash
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/full_smoke_1337
bash scripts/run_full_smoke_train_test.sh
```

Success ends with:

```text
FULL_SMOKE_PASS
AOSK_VARIANT=centerline-cldice-v1
SMOKE_TEST_SPLIT=smoke_test
CANONICAL_TEST_OPENED=NO
TEST_FIREWALL=CLOSED
```

The smoke-test split is derived from train/validation-domain data. It is not the canonical benchmark test.

## Three seeds

`run_all_seeds.sh` no longer silently defaults to N25. Choose the protocol explicitly:

```bash
export BASE_EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_3seed
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export NORMAL_FRACTION=0.0   # N0
# or: export NORMAL_FRACTION=0.25  # N25
bash scripts/run_all_seeds.sh
```

Seeds are exactly `1337`, `2027`, `31415`.

For paper-quality normal-supervision analysis, run N0 and N25 as separately identified protocols rather than describing N25 S0 as the historical crack-only control.

## Final test exactly once

Canonical test remains closed during training, smoke and validation.

After checkpoint, hyperparameters and validation threshold are frozen, create a `PROTOCOL_LOCK.json` binding checkpoint SHA, full-manifest SHA, dataset-content SHA and threshold, then run:

```bash
bash scripts/run_final_test.sh /absolute/path/PROTOCOL_LOCK.json
```

The runner atomically creates `.test_opened` before any canonical test image/mask is opened. There is no force/replay path for paper evaluation.

## CI and host acceptance

GitHub CI verifies installation, compileall, pytest and shell syntax. Implementation 2.0.2 passed GitHub Actions run #199 with 49/49 tests; the package built as version 2.0.2 and all canonical shell syntax checks passed on that code state.

GitHub CI does **not** certify the user's real `/hdd1/...` dataset or NVIDIA A30 runtime.

Final acceptance order:

```text
GitHub CI PASS
-> prepare_real_data.sh PASS on real files
-> A30/CUDA preflight PASS
-> critic qualification PASS
-> seed-1337 S0-S3 validation PASS
-> 3-seed N0 and/or N25 protocol
-> freeze method/hyperparameters
-> protocol lock
-> canonical test once
```

Never lower a data or critic gate after observing experiment outcomes.
