# OASIS-A2S v0.1.2 crack segmentation

Experimental research branch for **OASIS-A2S v0.1** with implementation revision **0.1.2**. The scientific OASIS core is unchanged from v0.1.1; v0.1.2 hardens transfer, calibration, provenance and evaluation.

```text
method_version          = OASIS-A2S-v0.1
implementation_revision = 0.1.2
package_version         = 0.1.2
branch                  = feat/oasis-a2s-v0.1.2
```

## Scientific core — unchanged

Binary crack Stage-I OASIS predicts:

```text
background / crack / fake
```

Training keeps the OASIS mechanisms used by the A2S hypothesis:

- per-pixel N+1 semantic discriminator objective;
- inverse-frequency class balancing;
- fake-class supervision for generated images;
- semantic adversarial generator supervision;
- class-aware LabelMix consistency;
- semantic + global latent noise conditioning in the training-only generator.

The discriminator remains a compact U-Net-style adaptation rather than a byte-for-byte copy of the archived upstream OASIS architecture. No Mamba, SAM, AOSK, physics simulator, extra critic, topology head or second decoder is added to the A2S core.

## v0.1.2 protocol

Development is strictly:

```text
FIT -> CAL -> VAL
```

- **FIT**: train A0 and OASIS Stage-I / A2-Full.
- **CAL**: select the crack-probability threshold from a frozen grid.
- **VAL**: report development metrics and paired per-image deltas. VAL does not retune the threshold.
- **TEST/final/holdout**: sealed by the development runner.

Default threshold grid:

```text
0.10, 0.15, ..., 0.90
```

Thresholds are calibrated independently for each arm on CAL and stored in deployment checkpoints. The evaluator never searches thresholds on external/final data.

## Arms

```text
A0       scratch 2-class supervised baseline
A1       direct Stage-I OASIS D using its two real semantic logits
A2-Full  exact D3 -> D2 transfer + full real-only fine-tuning
A2-WI    frozen weight interpolation between transferred pre-FT and A2-Full
```

For the current development protocol:

```text
A2-WI alpha = 0.8
```

This alpha is a **frozen development-discovered candidate**, not an optimization target and not a claimed novelty. Do not sweep alpha on VAL/final data.

## Training trajectories and resumability

Stage-I defaults to a 50-epoch maximum and saves:

```text
1 / 3 / 5 / 10 / 20 / 30 / 50
```

Stage-I checkpoints contain D/G weights, both optimizer states, history and RNG state. `--stage1-resume` continues from the saved optimizer state; optimizer-reset continuation is not canonical evidence.

A2-Full saves Stage-II trajectory checkpoints at:

```text
1 / 3 / 5 / 10 / 20 / 30
```

A0 receives a longer fixed default budget (`--a0-epochs 100`) so the scratch control is not intentionally under-trained. Epoch budgets remain protocol parameters and should be frozen before confirmatory runs.

## Metrics

v0.1.2 adds evaluation-only structural and reliability metrics without changing the Gate-1 training loss:

```text
Precision / Recall / Dice / IoU / Accuracy
mean per-image Dice
clDice
Boundary-F1 (1 px tolerance at evaluation resolution)
normal-image false-positive pixel fraction
paired per-image Dice bootstrap 95% CI
```

clDice/Boundary-F1 are **metrics only** in v0.1.2. They are not added to the training objective, preserving attribution to OASIS pretraining/transfer.

## Data and leakage preflight

Canonical runs hash dataset bytes, not only manifest text. Preflight requires FIT/CAL/VAL and by default enforces:

- non-empty FIT/CAL/VAL;
- `lineage_id` for every evidence row;
- no lineage crossing FIT/CAL/VAL;
- no exact RGB SHA256 duplicate crossing FIT/CAL/VAL;
- no duplicate canonical evidence row;
- matching original image/mask dimensions;
- valid normal-row/mask contracts.

The checkpoint records `dataset_content_sha256`, split counts and audit metadata. Diagnostic-only overrides are explicit (`--allow-missing-lineage`, `--allow-size-mismatch`).

## Git / reproducibility contract

Canonical runs require a resolvable Git HEAD and a clean worktree. Running from an unversioned ZIP fails closed unless `--allow-unversioned` is explicitly supplied for diagnostics. Dirty-tree runs also require an explicit diagnostic override.

Deterministic runs seed Python/NumPy/PyTorch, disable cuDNN benchmarking, enable deterministic algorithms and set the CUDA workspace configuration when applicable. Reproducibility claims remain scoped to a frozen environment; PyTorch does not promise bit-identical results across releases/platforms.

## Deployment

Deployment is one network only:

```text
RGB -> A1 Stage-I real logits or 2-class A0/A2 model -> frozen CAL threshold -> crack mask
```

The generator and optimizers are not accepted by deployment evaluation checkpoints.

## Canonical pilot command

```bash
python -m oasis_cycle_aosk.train_a2s \
  --manifest /path/to/manifest.jsonl \
  --out /path/to/a2s_v012_seed1337 \
  --fit-split fit \
  --cal-split cal \
  --val-split val \
  --size 256 \
  --batch 8 \
  --device cuda \
  --seed 1337 \
  --stage1-epochs 50 \
  --a0-epochs 100 \
  --stage2-epochs 30 \
  --wise-alpha 0.8
```

Do not use diagnostic overrides for canonical evidence.

## Decision policy

`results.json` reports **development signals**, not an automatic publish/continue gate. Primary paired comparisons are:

```text
A2-Full - A0
A2-WI   - A0
```

A fresh source-disjoint test must be used after method choices are frozen. Previously inspected debug/test packs are development diagnostics and must not be reused as final unbiased evidence.
