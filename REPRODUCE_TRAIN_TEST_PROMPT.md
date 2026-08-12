# Prompt to reproduce and audit OASIS-RC-v2

You are a senior computer-vision researcher, ML engineer and strict
reproducibility reviewer. Work only inside this `OASIS-RC-v2` folder.

## Provenance constraint

The exact historical v2 source was not preserved in the current workspace.
The included `train_oasis_rc_v2.py` and `losses_v2.py` are a reconstructed
canonical implementation of the audited v2 hypothesis. You must distinguish:

```text
historical v2 evidence -> reports/ and artifacts/
new reconstructed rerun -> code in src/ and new runs/
```

Never state that a new run exactly reproduces historical v2 numbers unless a
source hash proves that the historical source has been recovered.

## Research question

Does adding an explicit online corrupted-mask ranking reference make the
OASIS-RC critic useful to a same-backbone RGB-only crack segmentation student?
If not, reject v2.

## Fixed architecture

The deployment student is RGB-only. The training-only critic receives RGB and
one-channel soft masks. Relational fusion is:

```text
[F_image, F_mask, F_image * F_mask, abs(F_image - F_mask)]
```

It returns semantic background/crack/invalid logits, a mismatch map and a
pair-validity logit.

V2 compares three relations from the same image:

```text
E_gt        = E(D(RGB, GT))             detached
E_pred      = E(D(RGB, sigmoid(logit))) differentiable to student
E_corrupted = E(D(RGB, corrupt(GT)))    detached
```

The additional term is:

```text
softplus(E_pred - E_corrupted + margin)
```

It supplements GT ranking and the GT-background-only false-positive penalty.
All corruptions are generated online and are not written to a dataset.

## Immutable rules

1. Deployment is `RGB -> student -> mask`; no critic/AOSK/GAN at inference.
2. Threshold, checkpoint, lambda and architecture are selected without test.
3. Every v2 run has an identically initialized same-backbone control.
4. Use seeds `1337`, `2027`, `31415`; never select the best seed.
5. Block connected training when the critic gate fails.
6. `test_debug` is diagnostic and source-overlapping.
7. Do not describe OASIS-RC-v2 as a teacher.
8. Do not claim Q1/TIM/ECCV readiness without source-disjoint locked test,
   structural metrics and stable three-seed improvement.

## Environment and tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e .
python -m compileall -q src tests
pytest -q
```

The v2 test must verify that gradients reach student logits, critic parameters
stay frozen, corrupted references are detached and output contracts are valid.

## Data preparation

Raw images are not embedded. Remap the historical manifest if needed:

```bash
python scripts/remap_manifest.py \
  --input manifests/manifest_rebalanced_debug.jsonl \
  --output manifests/local_debug.jsonl \
  --old-prefix /workspace/scratch/4ab02dda3b35/pilot_real_debug/real_debug_data \
  --new-prefix /absolute/path/to/real_debug_data
```

Strict paper audit:

```bash
PYTHONPATH=src python -m oasis_cycle_aosk.audit \
  --manifest /absolute/path/to/source_disjoint_manifest.jsonl \
  --test-split test --require-source-disjoint
```

Stop if source/session/lineage leakage, missing normal test images, duplicate
pairs or image-mask misalignment is found.

## Smoke protocol

```bash
scripts/run_smoke.sh "$PWD/manifests/local_debug.jsonl" mobilenetv3
scripts/run_lightweight_smoke.sh "$PWD/manifests/local_debug.jsonl"
```

The smoke uses three epochs and validates implementation only. Do not rank
backbones or claim v2 improvement from it.

## Full three-seed protocol

Use 128x128, batch 4, AdamW `2e-4`, 12 epochs, BCE + Dice, warm-up 4 and ramp 3.
The frozen historical v2 candidate uses `lambda_oasis=0.003`, pair energy
weight `0.25` and corrupted ranking weight `1.0`:

```bash
scripts/run_three_seeds.sh /absolute/path/to/source_disjoint_manifest.jsonl mobilenetv3
```

Run lambda or pair-weight ablations only on validation. Required candidates:

```text
V0 control
V1 OASIS-RC v1 loss
V2 v2 lambda 0.003
V3 v2 lambda 0.010
V4 v2 lambda 0.030
V5 v2 pair_weight 1.0
V6 v2 without corrupted ranking
V7 v2 mask-shuffle control
```

Do not replace the mean with the best seed. Report paired confidence intervals
when the validation sample count supports them.

## Critic gate

```text
valid crack recall >= 0.80
invalid recall     >= 0.90
RGB pair drop      >= 0.05
mask pair drop     >= 0.05
```

The historical four-arm v2 smoke failed both pair-drop gates and correctly
blocked connected training. Preserve that failure in the record; do not lower
the gate after observing results.

## Evaluation and reporting

Use only validation-selected thresholds. Report pixel and structural metrics,
normal false alarms, source-wise statistics, parameters, FLOPs, CPU/edge-device
latency and memory. Checkpoint deployment must not contain critic/AOSK state.

Final report must separate:

1. historical v2 numbers;
2. reconstructed implementation results;
3. exact source gaps;
4. leakage limitations;
5. go/no-go decision.

If v2 fails, write:

```text
OASIS-RC-v2 rejected: explicit corrupted-mask ranking did not provide a
statistically credible gain over the paired RGB-only control.
```
