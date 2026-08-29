# OASIS-A2S v0.1 crack segmentation

Experimental research branch implementing **OASIS-A2S v0.1**: semantic-adversarial OASIS pretraining followed by transfer of the same discriminator into a deployable 2-class crack segmenter.

```text
method_version          = OASIS-A2S-v0.1
implementation_revision = 0.1.1
package_version         = 0.1.1
branch                  = feat/oasis-a2s-v0.1
```

This branch is intentionally narrow. It does **not** add OASIS-RC critics, AOSK, Mamba, SAM, physics simulation or topology heads to the A2S core.

## Scientific hypothesis

The Gate-1 question is deliberately simple:

```text
Does semantic-adversarial pretraining of an OASIS discriminator provide a
better initialization for real-data crack segmentation than training the same
2-class architecture from scratch?
```

The prototype evaluates three arms:

```text
A0  same 2-class discriminator architecture, supervised from scratch
A1  Stage-I OASIS N+1 discriminator, evaluated through its two real classes
A2  exact Stage-I D -> 2-class transfer, then real-only segmentation fine-tune
```

A0 and A2 use the same Stage-II architecture, loss, learning rate, epoch budget,
resolution, data view and shuffle seed. Their intended difference is the OASIS
semantic-adversarial initialization of A2.

## Architecture and training contract

For binary crack segmentation, Stage I predicts:

```text
background / crack / fake
```

The training-only generator is conditioned by the binary semantic map plus a
spatial noise tensor. Stage-I training uses the OASIS principles required by the
A2S hypothesis:

- per-pixel N+1 semantic discriminator objective;
- inverse-frequency class balancing for real semantic classes;
- fake-class supervision for generated images;
- semantic adversarial supervision for the generator;
- class-aware LabelMix consistency;
- semantic + 3D-noise conditioning in the generator.

Reference defaults for the Stage-I pilot are:

```text
lr_D             = 4e-4
lr_G             = 1e-4
lambda_labelmix  = 10
```

The discriminator is a compact U-Net-style adaptation, not a byte-for-byte copy
of the archived upstream OASIS architecture. OASIS-A2S reimplements the method
principles needed for the transfer experiment rather than vendoring the upstream
repository.

## Exact A2S transfer

Stage-I head:

```text
[BG, Crack, Fake]
```

Stage-II head:

```text
[BG, Crack]
```

`transfer_to_segmenter()` preserves the encoder, decoder, skip-path parameters,
and the BG/Crack classifier weights exactly; the Fake classifier row is removed.
The entire transferred segmenter is then fine-tuned on real image/mask pairs.

## Inference contract

Deployment is always:

```text
RGB -> transferred 2-class OASIS-A2S discriminator -> crack mask
```

The generator is training-only and must not be present in A0/A2 deployment
checkpoints. No critic, generator, discriminator wrapper or AOSK state is accepted
by the deployment evaluator.

## Development firewall

Development training rejects split names containing semantic tokens:

```text
test
final
holdout
```

Examples rejected include `test`, `final_external`, `test_2026`,
`external-final`, `holdout_v2` and `evaluation_test`.

The development evaluator is also manifest-provenance bound. By default:

```text
SHA256(evaluation manifest) == checkpoint.manifest_sha256
```

is mandatory. Development evaluation cannot override a manifest mismatch. A
future frozen external/final evaluation must opt in explicitly and the mismatch
and final-test overrides are recorded in the output artifact.

## Reproducibility policy

Canonical development runs enable deterministic PyTorch algorithms, disable
cuDNN benchmarking, enable deterministic cuDNN behavior, seed Python/NumPy/Torch,
and configure `CUBLAS_WORKSPACE_CONFIG` before CUDA seeding. Because PyTorch does
not guarantee identical results across releases, platforms, or CPU/GPU backends,
reproducibility claims are scoped to a frozen software/hardware execution setup.

Checkpoints record:

```text
method + implementation revision
git commit + dirty state
manifest SHA256
train/validation split names
seed, resolution and learning rates
epoch budgets and loss weights
PyTorch/CUDA/cuDNN/device determinism metadata
```

## Prototype command

```bash
python -m oasis_cycle_aosk.train_a2s \
  --manifest /path/to/manifest.jsonl \
  --out /path/to/experiments/a2s_gate1_seed1337 \
  --train-split train \
  --val-split val \
  --size 256 \
  --batch 8 \
  --device cuda \
  --seed 1337 \
  --stage1-epochs 30 \
  --stage2-epochs 30
```

Do not use `--allow-nondeterministic` for canonical evidence runs.

## Verification

Before real Gate-1 training, run from a clean checkout of the exact frozen commit:

```bash
python -m compileall src tests
pytest -q
```

Then run a small CPU or GPU end-to-end smoke using only train/validation rows.
Mechanical smoke success does not establish scientific efficacy.

The v0.1 Gate-1 decision is based on A0 versus A2. If A2 does not improve over A0
under the frozen pilot protocol, the OASIS-A2S direction should be treated as
negative/inconclusive rather than rescued by adding unrelated architecture.

## Historical OASIS-RC work

The parent branch `feat/oasis-rc-v2.1-dev3-q1` preserves the previous OASIS-RC-v2.1
experiments and documentation. This A2S branch intentionally separates the new
method hypothesis from that historical path.
