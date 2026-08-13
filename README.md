# OASIS-RC v2 crack-segmentation research package

This repository contains the current **strict reconstructed OASIS-RC v2** research implementation for crack segmentation. The exact historical v2 source snapshot was not preserved; historical numbers must not be described as bit-for-bit reproducible from this tree.

```text
experiment_id          = oasis-rc-v2-relational-hard-negative
checkpoint_schema      = 2
method_version         = OASIS-RC-v2
implementation_version = 2.0.1
```

Method-specific critic, corruptions, losses, checkpoint rules, qualification gates and protocol code live under `src/oasis_rc_v2/`. RGB students, shared manifest loading and the current AOSK ablation live under `src/oasis_cycle_aosk/`.

## Scientific contract

Controlled arms:

```text
S0 Control                    L = L_seg
S1 OASIS-RC v2                L = L_seg + lambda_oasis * rc_ramp * L_RCv2
S2 AOSK-Oriented              L = L_seg + lambda_aosk * L_AOSK_oriented
S3 OASIS-RC v2 + AOSK-Oriented
                              L = L_seg + lambda_oasis * rc_ramp * L_RCv2
                                        + lambda_aosk * L_AOSK_oriented
```

`L_seg = BCE + Dice`. The current AOSK implementation is explicitly identified as `oriented-consistency-v1`; it must not be described as a skeleton-aware implementation unless a separate skeleton-aware method is restored and versioned.

OASIS-RC v2 is training-only. Deployment is always:

```text
RGB -> student -> crack logits/mask
```

Student checkpoints are rejected if they contain critic, generator, discriminator or AOSK training state.

## OASIS-RC v2 critic

The critic consumes `(RGB, soft_mask)` with separate image/mask encoders and relational fusion:

```text
fi, fm, fi*fm, abs(fi-fm)
```

Outputs:

```text
semantic : valid background / valid crack / invalid RGB-mask relation
mismatch : pixel mismatch logit
pair     : pair-validity logit
```

Online hard negatives:

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

C7/C8 never silently fall back to another corruption while keeping a donor label. C7 requires a genuine non-self crack donor. C9 uses RGB texture when RGB is supplied. Corruptions are online only, no-op corruptions are rejected, `torch.roll` is not used, and changed-area/IoU constraints are enforced.

Critic objective:

```text
L_critic =
    L_semantic_balanced
  + lambda_crack    * L_valid_crack_dice
  + lambda_mismatch * L_mismatch
  + lambda_pair     * L_pair
  + lambda_rgb_mask * L_RGB-mask-shuffle
```

RGB spatial shuffle is a dedicated pair-validity negative and does not reward semantic prediction from an unchanged mask.

Student RC objective remains the reconstructed v2 lineage:

```text
rank(pred, GT)
+ rank(pred, corrupted)
+ background false-positive penalty
```

GT/corrupted reference energies are detached and the critic is frozen during student training.

## Data integrity and test firewall

The canonical data path is fail-closed:

```text
canonical manifest
  -> split/lineage normalization
  -> cross-split leakage repair (test > val > train)
  -> exact same-split RGB/pair deduplication
  -> fail-closed CleanEval
  -> row-level N0 certification only
  -> normal-RGB corrupt/duplicate/cross-label audit
  -> full manifest
  -> full Gate 0
  -> N25 train/val view + Gate 0 certificate
  -> N0  train/val view + Gate 0 certificate
  -> trainer
```

Gate-0 certificates bind both:

```text
manifest SHA256
actual referenced image/mask bytes (dataset_content_sha256)
```

An inventory JSONL with row-level image/mask hashes is emitted next to each Gate-0 certificate. If an image or mask changes in place after certification, the trainer refuses the certificate.

Official trainers refuse any manifest containing `split=test`.

## Prepare real data

```bash
cd /hdd1/hieulc/Oasis_AOSK/OASIS-RC-v2

export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export DATA_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_real/data
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb

bash scripts/prepare_real_data.sh
```

Success must end with `REAL_DATA_READY` and create at least:

```text
manifest_full_with_normal.jsonl
gate0_full.json

manifest_trainval_with_normal.jsonl
gate0_training.json

manifest_trainval_n0.jsonl
gate0_training_n0.json

normal_audit/summary.json
```

Do not train if data preparation fails.

## CUDA reproducibility policy

Canonical GPU runners default to:

```text
DETERMINISM_MODE=best_effort
CUBLAS_WORKSPACE_CONFIG=:4096:8
cuDNN benchmark=false
cuDNN deterministic=true
TF32=false
```

`best_effort` asks PyTorch for deterministic algorithms but warns instead of aborting when the installed CUDA/PyTorch stack has an operation without a strict deterministic implementation. `strict` remains available for hardware/software stacks that support the complete graph.

Every run records Python, PyTorch, CUDA, cuDNN, GPU model, compute capability, Git SHA and determinism mode in runtime metadata.

## One-seed real-data acceptance run

Start with seed `1337`:

```bash
export EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_real/seed_1337
export DATA_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_real/data
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python

export SEED=1337
export NORMAL_FRACTION=0.25
export CRITIC_EPOCHS=10
export EPOCHS=12
export LAMBDA_OASIS=0.001
export LAMBDA_AOSK=0.01
export DETERMINISM_MODE=best_effort

bash scripts/run_training_ready.sh 2>&1 | tee "$EXP_ROOT/training_ready.log"
```

For the N0 ablation:

```bash
export NORMAL_FRACTION=0.0
bash scripts/run_training_ready.sh
```

The runner automatically selects the N0 crack-only manifest/certificate when `NORMAL_FRACTION=0`; otherwise it selects the normal-augmented training view.

## Critic qualification

Before connected S1/S3 are allowed to train, the critic must satisfy the frozen gate.

Base requirements include:

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

For normal-supervised runs, additional requirements are:

```text
valid_normal_bg_recall  >= 0.80
normal_pair_valid_mean  >= 0.50
normal_invalid_rate     <= 0.20
normal_samples          > 0
C8 samples              > 0
```

Per-corruption recall is emitted for C1-C9. C7 is measured only where a genuine non-self crack donor exists. If the validation data cannot produce a required corruption, the gate fails instead of fabricating a negative.

The normal diagnostic currently uses `normal_train`; it is a critic qualification diagnostic, not independent held-out evidence of normal-image generalization.

## Validation cost

Threshold selection performs **one model forward per validation batch**, caches the batch probabilities in memory for the current batch and sweeps thresholds tensor-wise. It does not rerun the network once per threshold.

The selected threshold is stored in the student checkpoint as `threshold_validation`.

## Four validation arms

Official output directories are:

```text
S0_control
S1_oasis_rc_v2
S2_aosk_oriented
S3_oasis_rc_v2_aosk_oriented
```

S0/S1/S2/S3 share the same seed-specific student initialization. S1/S3 share the same frozen qualified critic. `lambda_oasis` and `lambda_aosk` are explicit runner environment variables and are written into checkpoint metadata.

Validation-only evaluation:

```bash
export EXP_ROOT=/path/to/seed_1337
export MANIFEST=/path/to/data/manifest_trainval_with_normal.jsonl
export ARM_ROOT="$EXP_ROOT/arms"

bash scripts/run_validation_eval.sh
```

Missing official arm checkpoints are a hard failure. The validation evaluator rejects any manifest containing canonical test rows.

## Three seeds

After seed 1337 passes the real-data acceptance run:

```bash
export BASE_EXP_ROOT=/hdd1/hieulc/Oasis_AOSK/experiments/oasis_rc_v2_3seed
export DATA_ROOT="$BASE_EXP_ROOT/data"
export CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/absolute/path/to/true_normal_rgb
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export NORMAL_FRACTION=0.25
export CRITIC_EPOCHS=10
export EPOCHS=12
export LAMBDA_OASIS=0.001
export LAMBDA_AOSK=0.01
export DETERMINISM_MODE=best_effort

bash scripts/run_all_seeds.sh
```

Seeds are exactly:

```text
1337
2027
31415
```

Canonical test remains closed throughout training and validation.

## Final test: exactly once after protocol lock

`evaluate_rc.py --split test` cannot be called directly without a final-test authorization marker. The legacy `evaluate_checkpoint.sh` also refuses `split=test`.

After model, hyperparameters and validation threshold are frozen, create `PROTOCOL_LOCK.json` containing at least:

```json
{
  "selected_checkpoint": "/absolute/path/student_only.pt",
  "selected_checkpoint_sha256": "...",
  "manifest": "/absolute/path/manifest_full_with_normal.jsonl",
  "manifest_sha256": "...",
  "dataset_content_sha256": "...",
  "threshold": 0.42,
  "hyperparameters_locked": true,
  "output": "/absolute/path/final_test.json"
}
```

`dataset_content_sha256` comes from the final full-benchmark Gate-0 certificate. `threshold` must equal the selected checkpoint's `threshold_validation`.

Run:

```bash
bash scripts/run_final_test.sh /absolute/path/PROTOCOL_LOCK.json
```

The runner atomically creates `PROTOCOL_LOCK.json.test_opened` **before any test image/mask is opened**. If evaluation crashes afterward, the marker remains and a repeated canonical-test opening is refused. There is no `FORCE_FINAL_TEST` paper path.

## Reproducibility artifacts

Critic/student checkpoints bind or record:

```text
method / implementation identity
manifest SHA256
dataset-content SHA256
Gate-0 certificate SHA256
student-init SHA256
critic SHA256 where applicable
training hyperparameters
selected validation threshold
runtime Python/PyTorch/CUDA/cuDNN/GPU/Git metadata
```

Student histories separately log segmentation, RC, weighted RC, AOSK, weighted AOSK and relation-energy diagnostics.

## CI and real-host acceptance

GitHub CI checks installation, compileall, pytest and shell syntax. Regression tests include a real-file CPU pipeline smoke using actual PNG images/masks and explicit tests for dataset-byte binding, threshold-forward count, final-test authorization, C1-C9 corruption semantics and critic gates.

GitHub CI cannot certify the user's `/hdd1/...` dataset or actual A30 GPU. The final acceptance step before claiming experimental results is therefore:

```text
prepare_real_data.sh PASS on real files
GPU environment preflight PASS
critic qualification PASS
seed-1337 S0-S3 validation PASS
then 3-seed validation
then protocol lock
then canonical test once
```

Never lower a data or critic gate after seeing experiment outcomes. Any scientific change to the reconstructed RC ranking objective should receive a new method/version identifier rather than silently changing OASIS-RC v2.
