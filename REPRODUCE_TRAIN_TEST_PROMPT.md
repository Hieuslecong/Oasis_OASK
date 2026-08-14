# Current reproduction and controlled-experiment protocol

> **Status:** this file describes the repaired `OASIS-RC-v2-reconstructed`
> implementation. The exact historical v2 source is missing; see
> `HISTORICAL_SOURCE_GAP.md`. Do not describe new runs as exact historical
> reproduction.

## Scientific contract

```text
S0 Control        L = L_seg
S1 OASIS-RC-v2    L = L_seg + lambda_oasis * rc_ramp * L_RCv2
S2 AOSK           L = L_seg + lambda_aosk * L_AOSK
S3 OASIS + AOSK   L = L_seg + lambda_oasis * rc_ramp * L_RCv2
                             + lambda_aosk * L_AOSK
```

AOSK is active independently of RC warmup/ramp. Critic/AOSK are training-only.
Deployment is RGB-only student inference.

## Mandatory order

1. Checkout the exact branch/commit to be tested.
2. Install `requirements-tested.txt` and package editable.
3. `compileall` + `pytest` must pass.
4. Build/merge the true-normal manifest without altering canonical val/test.
5. Run Gate 0 at the effective training resolution.
6. Create one exact student-init checkpoint per seed/backbone.
7. Train one critic per seed; S1 and S3 must share the exact critic checkpoint.
8. Run validation-only micro-smoke S0/S1/S2/S3.
9. Run auxiliary-gradient diagnostics.
10. Freeze normal fraction/lambdas/epochs/threshold procedure on validation.
11. Run paired seeds 1337, 2027, 31415.
12. Open canonical test only once after the protocol is frozen.

## Environment verification

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
python -m compileall src scripts tests
pytest -vv
```

## Prepare certified real data

Safe default when parent identity cannot yet be recovered:

```bash
DATA_ROOT=/ABSOLUTE/PATH/data \
CANONICAL_MANIFEST=/ABSOLUTE/PATH/canonical_manifest.jsonl \
NORMAL_ROOT=/ABSOLUTE/PATH/audited_normal_rgb \
bash scripts/prepare_real_data.sh
```

Do **not** create `normal_val` by randomly splitting patches. If filenames encode
strongest parent/session identity, pass a reviewed `--lineage-regex` and then a
ratio below 1.0. The first capturing group becomes the lineage key.

## Gate 0

```bash
The preparation script emits both a full-benchmark certificate and a linked
train/validation-view certificate. Training rejects either a missing link or a
changed manifest/data payload.
```

Stop on any failure. Important checks include raw/decoded RGB and mask hashes,
decoded pair hashes, split-independent lineage, explicit normal semantics,
native/resized foreground, and alignment certification for mismatched native
image/mask resolutions.

`alignment_verified=true` must only be added after a GT-only spatial audit. It
must never be inserted merely to bypass Gate 0.

## Canonical student initialization

```bash
python scripts/create_student_init.py \
  --seed 1337 \
  --student-kind multiscale \
  --student-width 16 \
  --out /ABSOLUTE/PATH/student_init_seed1337.pt
```

Official student runs fail without `--student-init-checkpoint`. Connected runs
fail without a frozen `--critic-checkpoint`. The debug escape hatches are not
paper protocol.

## Four-arm GPU micro-smoke

```bash
scripts/run_smoke.sh \
  /ABSOLUTE/PATH/manifest_trainval_with_normal.jsonl \
  /ABSOLUTE/PATH/gate0_training.json \
  /ABSOLUTE/PATH/student_init_seed1337.pt \
  multiscale \
  /ABSOLUTE/PATH/gate0_full.json
```

This is validation-only and must not be reported as scientific evidence.

## Gradient diagnostics

```bash
python scripts/diagnose_aux_gradients.py \
  --config configs/canonical_gpu_256_seed1337.yaml \
  --manifest /ABSOLUTE/PATH/manifest_with_normal.jsonl \
  --gate0-certificate /ABSOLUTE/PATH/gate0_training.json \
  --full-gate0-certificate /ABSOLUTE/PATH/gate0_full.json \
  --student-init-checkpoint /ABSOLUTE/PATH/student_init_seed1337.pt \
  --critic-checkpoint /ABSOLUTE/PATH/critic.pt \
  --normal-fraction 0.25 \
  --batches 50 \
  --out /ABSOLUTE/PATH/gradient_diagnostics.json
```

Use weighted gradient ratios and cosine alignment before changing lambdas. Do
not infer gradient influence from raw loss magnitude alone.

## Paired three-seed validation

```bash
NORMAL_FRACTION=0.0 \
CANONICAL_MANIFEST=/ABSOLUTE/PATH/canonical_manifest.jsonl \
BASE_EXP_ROOT=/ABSOLUTE/PATH/three_seed_n0 \
bash scripts/run_all_seeds.sh
```

The script creates paired init checkpoints and one shared critic per seed, then
runs S0/S1/S2/S3. It does not evaluate test.

## Final test

Only after validation choices are frozen:

```bash
bash scripts/run_final_test.sh /ABSOLUTE/PATH/PROTOCOL_LOCK.json
```

Never tune lambda, checkpoint, resolution, normal fraction, architecture or
threshold from this output.

## Reporting

Report at minimum Dice/F1, IoU, precision, recall, paired multi-seed differences,
critic gate metrics, auxiliary-gradient diagnostics, normal false-positive
metrics when a lineage-safe normal_val exists, parameters/latency/memory, exact
git/runtime provenance, manifest/init/critic/checkpoint SHA256 values, and all
known data limitations.

Do not claim Q1/TIM readiness from a smoke run or a single favorable seed.
