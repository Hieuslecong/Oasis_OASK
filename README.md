# OASIS-RC-v2 crack-segmentation research package

This repository contains a **reconstructed** OASIS-RC-v2 training-time
relational critic for RGB-only crack segmentation plus the current controlled
AOSK ablation. The exact historical v2 source was not preserved; see
`HISTORICAL_SOURCE_GAP.md`. Historical metrics must not be described as
bit-for-bit reproducible from the current code.

## Current scientific contract

Four arms use the same RGB-only student, canonical initialization, manifest,
data order policy, augmentation RNG, optimizer and validation split:

```text
S0 Control        L = L_seg
S1 OASIS-RC-v2    L = L_seg + lambda_oasis * rc_ramp * L_RCv2
S2 AOSK           L = L_seg + lambda_aosk * L_AOSK
S3 OASIS + AOSK   L = L_seg + lambda_oasis * rc_ramp * L_RCv2
                             + lambda_aosk * L_AOSK
```

AOSK is independent of RC warmup/ramp. Critic and AOSK are training-only.
Deployment remains:

```text
RGB -> student -> crack logits/mask
```

## True-normal RGB support

External non-cracked wall images can be added as `normal_train` with a virtual
zero mask. They are never silently appended to canonical validation/test.
If a `normal_val` split is created, it must be split by strongest available
parent/session lineage using `--lineage-regex`; otherwise keep all external
normal RGB in train.

Example:

```bash
python scripts/add_normal_rgb_to_manifest.py \
  --canonical-manifest /path/to/frozen_manifest.jsonl \
  --normal-root /hdd1/hieulc/Oasis_AOSK/datasets/structural_defects/Walls/Non-cracked \
  --train-ratio 1.0 \
  --out /path/to/manifest_with_normal.jsonl
```

If parent identity is known from filenames, create an auxiliary normal-val
split safely:

```bash
python scripts/add_normal_rgb_to_manifest.py \
  --canonical-manifest /path/to/frozen_manifest.jsonl \
  --normal-root /path/to/Walls/Non-cracked \
  --lineage-regex '<REGEX_WITH_PARENT_CAPTURE_GROUP>' \
  --train-ratio 0.90 \
  --out /path/to/manifest_with_normal.jsonl
```

Do not use `--allow-file-level-lineage` for paper evidence.

## Gate 0

Use the effective training resolution:

```bash
PYTHONPATH=src python -m oasis_cycle_aosk.audit \
  --manifest /path/to/manifest_with_normal.jsonl \
  --resize-size 256 \
  --normal-policy train
```

Gate 0 checks, among other things:

- literal `is_normal` semantics;
- missing/cracked masks;
- raw and decoded RGB duplicates;
- raw and decoded binary-mask duplicates;
- decoded RGB-mask pair duplicates;
- split-independent lineage;
- native image/mask geometry;
- explicit `alignment_verified=true` for different native resolutions;
- cracks that disappear after resize.

Do not train if Gate 0 fails.

## Canonical student initialization

Create one initialization per seed/backbone and reuse the exact file across all
four arms:

```bash
python scripts/create_student_init.py \
  --seed 1337 \
  --student-kind multiscale \
  --student-width 16 \
  --out /path/to/student_init_seed1337.pt
```

Official student runs require `--student-init-checkpoint`. Connected S1/S3 runs
also require the **same frozen critic checkpoint**. The debug escape hatches
`--allow-random-init` and `--allow-inline-critic` are not for controlled paper
experiments.

## GPU micro-smoke

```bash
scripts/run_smoke.sh \
  /path/to/manifest_with_normal.jsonl \
  /path/to/student_init_seed1337.pt \
  multiscale
```

This trains one critic and reuses it for S1/S3, then runs S0/S1/S2/S3 for two
validation-only epochs. It does **not** evaluate the canonical test split.

## Gradient diagnostics

Before increasing auxiliary weights, measure weighted gradient strength and
alignment:

```bash
python scripts/diagnose_aux_gradients.py \
  --config configs/canonical_gpu_256_seed1337.yaml \
  --manifest /path/to/manifest_with_normal.jsonl \
  --student-init-checkpoint /path/to/student_init_seed1337.pt \
  --critic-checkpoint /path/to/critic.pt \
  --normal-fraction 0.25 \
  --batches 50 \
  --out /path/to/gradient_diagnostics.json
```

The diagnostic reports gradient norms/ratios/cosines plus `e_pred`, `e_gt`,
`e_corrupted` and relation-energy deltas. It uses a fixed eval-mode student and
does not update model weights.

## Three-seed validation protocol

After Gate 0, micro-smoke and gradient diagnostics pass:

```bash
scripts/run_three_seeds.sh /path/to/frozen_manifest.jsonl multiscale
```

Seeds are `1337`, `2027`, `31415`. The script runs S0/S1/S2/S3 with paired
initialization and a shared critic per seed. It does not evaluate test.
Hyperparameters must be frozen from validation before the single final test.

## Evaluation

`evaluate_rc.py` loads student-only checkpoints, rejects training-only state and
uses the training resolution stored in the checkpoint unless an explicit
resolution ablation is requested.

```bash
scripts/evaluate_checkpoint.sh \
  student_only.pt manifest.jsonl test <VALIDATION_THRESHOLD> result.json
```

Run this on the canonical test exactly once after the protocol is frozen.

## Reproducibility

Each student run writes:

- `student_only.pt`
- `history.json`
- `validation.json`
- optional `normal_validation.json`
- `effective_config.json`
- `run_metadata.json`

Metadata records git/runtime provenance, exact command, manifest SHA256,
student-init SHA256 and critic-checkpoint SHA256. Generate source hashes at run
time:

```bash
python scripts/write_source_hashes.py --out /path/to/run/source_hashes.txt
```

## Current evidence status

Historical v2 evidence is negative/near-tied with control. The current normal-RGB
repair is intended to test whether true-normal supervision restores useful
false-positive suppression under a controlled protocol. A negative result is
valid evidence; do not tune on test or lower critic gates after observing
results.
