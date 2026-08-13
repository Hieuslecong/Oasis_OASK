# OASIS-RC v2 crack-segmentation research package

This repository contains the current **strict reconstructed OASIS-RC v2** research implementation for crack segmentation. The original historical v2 source was not preserved, so historical numbers must not be described as bit-for-bit reproducible. The current method contract is frozen as:

```text
experiment_id          = oasis-rc-v2-relational-hard-negative
checkpoint_schema      = 2
method_version         = OASIS-RC-v2
implementation_version = 2.0.0
```

Method-specific critic, corruptions, losses, checkpoint rules, qualification gates and protocol code live under `src/oasis_rc_v2/`. Shared RGB students, data loading and AOSK remain in `src/oasis_cycle_aosk/`.

## Scientific contract

The four controlled arms are:

```text
S0 Control        L = L_seg
S1 OASIS-RC v2    L = L_seg + lambda_oasis * rc_ramp * L_RCv2
S2 AOSK           L = L_seg + lambda_aosk * L_AOSK
S3 OASIS + AOSK   L = L_seg + lambda_oasis * rc_ramp * L_RCv2
                             + lambda_aosk * L_AOSK
```

`L_seg = BCE + Dice`. OASIS-RC v2 uses ranking(pred, GT) + ranking(pred, corrupted) + a background false-positive penalty. The critic is frozen during student training; GT/corrupted reference energies are detached; RC uses warm-up/ramp. AOSK is training-only and independent of the RC schedule.

Deployment is always:

```text
RGB -> student -> crack logits/mask
```

No critic, generator, discriminator or AOSK state is allowed in a deployment checkpoint.

## OASIS-RC v2 critic

The critic consumes `(RGB, soft_mask)` and has separate image/mask encoders, relational fusion (`fi`, `fm`, `fi*fm`, `abs(fi-fm)`), and three outputs:

```text
semantic : background / crack / invalid RGB-mask relation
mismatch : pixel mismatch logit
pair     : pair-validity logit
```

The online hard-negative suite is C1-C9:

```text
C1 translation
C2 erosion
C3 dilation
C4 local crack break
C5 wrong width
C6 wrong connection / bridge
C7 non-self donor mask
C8 crack mask on true-normal RGB
C9 texture false-positive blob
```

Corruptions are regenerated/fixed if they are no-ops, must satisfy a changed-area/IoU qualification, never use circular `torch.roll`, and donor masks must be non-self. RGB spatial shuffle is supervised as the dedicated pair-validity negative `L_RGB-mask-shuffle`; it is not allowed to reward semantic prediction from the unchanged mask.

The critic objective is:

```text
L_critic =
    L_semantic_balanced
  + lambda_crack    * L_valid_crack_dice
  + lambda_mismatch * L_mismatch
  + lambda_pair     * L_pair
  + lambda_rgb_mask * L_RGB-mask-shuffle
```

## Strict data and test firewall

**Official trainers never receive canonical test rows.** Dataset certification is a separate process:

```text
canonical manifest
  -> strongest-lineage leakage repair
  -> fail-closed CleanEval
  -> normal-RGB audit/quarantine
  -> full manifest (train/val/test + normal_train)
  -> full Gate 0 certificate
  -> exact train/val projection (test removed)
  -> training-view Gate 0 certificate
  -> trainer
```

The trainer refuses a manifest containing `split=test` and requires a SHA-bound `training_view` Gate-0 certificate. Canonical test remains available only through the final protocol-lock entrypoint.

Normal-RGB preparation audits corrupt files, raw/decoded duplicates and exact cross-label duplicates against canonical cracked RGB. Problematic normal files are excluded from the **derived manifest only**; raw data is never deleted or modified.

## Environment check on the GPU host

```bash
nvidia-smi

/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Official GPU configs are provided for seeds `1337`, `2027`, and `31415` at 256x256.

## Canonical real-data preparation only

Set the canonical Gate-0 input manifest and true-normal RGB directory, then run:

```bash
cd /path/to/OASIS-RC-v2

export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export DATA_ROOT=/path/to/experiment/data
export CANONICAL_MANIFEST=/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/path/to/true_normal_rgb

bash scripts/prepare_real_data.sh
```

The command must end with `REAL_DATA_READY` and create:

```text
$DATA_ROOT/manifest_full_with_normal.jsonl
$DATA_ROOT/manifest_trainval_with_normal.jsonl
$DATA_ROOT/gate0_full.json
$DATA_ROOT/gate0_training.json
$DATA_ROOT/normal_audit/summary.json
```

Do not train if either Gate-0 certificate is not `PASS`.

## Canonical one-seed training

The single supported end-to-end entrypoint is:

```bash
cd /path/to/OASIS-RC-v2

export EXP_ROOT=/path/to/experiment/seed_1337
export DATA_ROOT=/path/to/experiment/data
export CANONICAL_MANIFEST=/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/path/to/true_normal_rgb
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export SEED=1337
export NORMAL_FRACTION=0.25
export CRITIC_EPOCHS=10
export EPOCHS=12

bash scripts/run_training_ready.sh 2>&1 | tee "$EXP_ROOT/training_ready.log"
```

This performs/validates data preparation, creates the seed-specific canonical student initialization, trains the critic, applies the critic qualification gate, and only then runs validation-only S0/S1/S2/S3. It finishes with:

```text
TRAINING_PIPELINE_READY seed=1337
TEST_FIREWALL=CLOSED
```

The critic must satisfy at least:

```text
valid_crack_recall >= 0.80
invalid_recall     >= 0.90
rgb_pair_drop      >= 0.05
mask_pair_drop     >= 0.05
rgb_pair_samples   > 0
mask_pair_samples  > 0
no background-only collapse
```

A failed critic gate blocks connected student arms. Do not lower the gate after seeing results.

## Canonical three-seed protocol

After the real-data preparation path is confirmed, run:

```bash
export BASE_EXP_ROOT=/path/to/experiment/oasis_rc_v2_3seed
export DATA_ROOT="$BASE_EXP_ROOT/data"
export CANONICAL_MANIFEST=/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/path/to/true_normal_rgb
export PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export NORMAL_FRACTION=0.25
export CRITIC_EPOCHS=10
export EPOCHS=12

bash scripts/run_all_seeds.sh
```

The runner executes seeds `1337`, `2027`, `31415`. S0/S1/S2/S3 share the seed-specific initialization; S1/S3 share the same frozen qualified critic for that seed. Canonical test remains closed.

## Validation evaluation

Use only the certified train/val manifest:

```bash
export EXP_ROOT=/path/to/experiment/seed_1337
export MANIFEST=/path/to/experiment/data/manifest_trainval_with_normal.jsonl
export ARM_ROOT="$EXP_ROOT/arms"

bash scripts/run_validation_eval.sh
```

The script refuses a manifest containing `split=test`.

## Final test — exactly once after lock

After architecture, weights, checkpoint selection and validation threshold are frozen, create a `PROTOCOL_LOCK.json` containing at least:

```json
{
  "selected_checkpoint": "/absolute/path/student_only.pt",
  "selected_checkpoint_sha256": "...",
  "manifest": "/absolute/path/manifest_full_with_normal.jsonl",
  "manifest_sha256": "...",
  "hyperparameters_locked": true,
  "output": "/absolute/path/final_test.json"
}
```

Then run exactly once:

```bash
bash scripts/run_final_test.sh /path/to/PROTOCOL_LOCK.json
```

The runner verifies checkpoint/manifest SHA256 and creates a `.test_opened` marker to prevent accidental repeated opening. Do not use `FORCE_FINAL_TEST=1` for paper evidence.

## CI and regression coverage

CI runs package installation, Python compile checks, pytest and shell syntax checks. The suite includes a real-file end-to-end CPU smoke that creates actual PNG images/masks and exercises:

```text
leakage repair
-> CleanEval
-> normal-RGB audit
-> full Gate 0
-> train-view Gate 0
-> one critic optimizer epoch
-> one student optimizer epoch
-> checkpoint/validation artifacts
```

This proves the file-backed pipeline reaches optimizer/checkpoint code without requiring the real `/hdd1/...` dataset. The final acceptance step for a real experiment is still the authoritative Gate 0 and critic qualification on the actual GPU host/data.

## Reproducibility rule

Never tune on canonical test. Never lower data/critic gates after seeing experiment outcomes. If a method change is introduced after the protocol is frozen, give it a new method/version identifier and rerun the affected controlled arms.
