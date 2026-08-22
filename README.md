# OASIS-RC-v2.1 crack segmentation

Compact research implementation of **OASIS-RC-v2.1-dev3** for training lightweight RGB-only crack-segmentation models with training-only relational and orientation-aware supervision.

```text
method_version         = OASIS-RC-v2.1
implementation_version = 2.1.0-dev3
checkpoint_schema      = 5
experiment_id          = oasis-rc-v2.1-gt-anchored-relational-energy-head
trainer_contract       = oasis-rc-v21-canonical-v1
```

Scientific source of truth: `METHOD_SPEC_V2_1.md`  
Metric contract: `METRIC_SPEC_V2_1.md`  
Executable protocol: `protocols/real_data_v21.json`

## Inference contract

Deployment is always:

```text
RGB -> lightweight student -> crack logits/mask
```

The relation critic, relation-energy head, AOSK and clDice supervision are training-only. Deployment checkpoints must not contain critic/generator/discriminator/AOSK state.

The canonical primary student is MobileNetV3-Small-style with **fixed canonical width metadata `16`**. This implementation does not expose width scaling for MobileNetV3; canonical shared initialization rejects any other `student_width`, and deployment-checkpoint validation rejects non-16 MobileNetV3 metadata. Other lightweight backbones retain their explicit width parameter.

## Experimental arms

```text
B0  BCE + Dice
B1  B0 + clDice
B2  B0 + frozen pretrained pair-critic BCE
S1  B0 + OASIS-RC-v2.1
S2  B0 + AOSK structure-tensor-v2
S3  B0 + OASIS-RC-v2.1 + AOSK structure-tensor-v2
```

B2 is a frozen pair-critic ablation, not conventional jointly-trained adversarial learning. Primary paired contrasts are `B1-B0`, `B2-B0`, `S1-B0`, `S2-B0`, and `S3-S2`. Null/negative effects are valid outcomes.

## Data protocols

**N0:** no external normal supervision; certified native-empty rows may remain internal true negatives.

**N25:** external true-normal RGB occupies 25% of the training batch budget. Normal data must be lineage-disjoint across `normal_train`, `normal_val`, and `normal_test`. Critic qualification for N25 uses held-out `normal_val`, never `normal_train` as a fallback. Dev3 additionally requires relation-energy PASS for both C9 texture-guided false positives and C8 crack-donor false positives on held-out normal RGB.

Prepare the canonical benchmark and training views with:

```bash
export DATA_ROOT=/path/to/data_v21
export CANONICAL_MANIFEST=/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/path/to/true_normal_rgb
export LINEAGE_REGEX='...'
bash scripts/prepare_real_data_v21.sh
```

The preparation path performs cleaning/CleanEval, audited normal splitting, full-benchmark Gate0, and separate N0/N25 training-view Gate0 certificates. Canonical `test` and `normal_test` rows never enter training views.

## Development run

Canonical order:

```text
Gate0
-> CUDA preflight
-> shared random student initialization
-> B0/S0 baseline
-> critic training
-> representation + crack/normal relation-energy qualification
-> trained-S0 RC gradient/energy diagnostic
-> freeze auxiliary weights
-> B0/B1/B2/S1/S2/S3
-> separate crack-val and normal-val evaluation
```

Main entry point:

```bash
export NORMAL_FRACTION=0.0   # N0, or 0.25 for N25
export SEED=1337
export EXP_ROOT=/path/to/experiments/v21/seed1337
export DATA_ROOT=/path/to/data_v21
export CANONICAL_MANIFEST=/path/to/canonical_manifest.jsonl
export NORMAL_ROOT=/path/to/true_normal_rgb
export LINEAGE_REGEX='...'
bash scripts/run_training_ready_v21.sh
```

The dev3 student default is **100 epochs** with best-validation checkpoint selection. This is a development budget, not a claimed optimum; the confirmatory budget must be frozen after convergence is inspected on development seed 1337 only. All paired arms then use the same frozen budget.

Development critic gates are fail-closed. At minimum: valid-crack recall `>=0.80`, invalid recall `>=0.90`, RGB/mask pair drops `>=0.05`, minimum required-corruption recall `>=0.70`, at least 16 samples per required corruption, positive energy-gap fraction `>=0.70`, continuous path-order fraction `>=0.65`, mean/median energy gap `>0`, at least 16 energy samples, and finite energies. N25 applies the same energy criteria to C9 normal-texture and C8 normal-donor trajectories. Do not lower gates to obtain a PASS.

## Evaluation and statistics

Crack-positive rows use precision, recall, Dice, IoU, clDice, skeleton precision/recall and component-excess metrics. True-negative rows are evaluated separately with normal false-positive pixels/components and any-FP rate; crack-overlap metrics are not averaged over empty targets.

Confirmatory inference uses the training seed as the sampling unit. Canonical confirmatory seeds are:

```text
2027  31415  42421  51511  62617
```

The immutable final bundle requires all six arms for all five seeds, frozen thresholds, exact data/spec/protocol/evaluator hashes, paired initialization/training-view provenance and one frozen Git commit. Exact sign-flip p-values remain secondary because five paired seeds give a coarse discrete null distribution.

## Verification policy

**Testing is local/external; GitHub is source storage only.** GitHub is not used as the scientific execution environment.

Before source is stored, run local checks in the execution environment, for example:

```bash
python -m compileall src scripts tests
pytest -q
```

Then run an end-to-end smoke on representative train/validation data with the canonical test firewall closed. Smoke evidence is mechanical integration evidence only; it does not establish model efficacy.

For real experiments, `scripts/preflight_v21_real_gpu.py` remains the target-host CUDA/backward gate. Full scientific GO additionally requires the real dataset Gate0 certificates, real critic qualification, trained-S0 diagnostic, frozen development decisions and the complete multi-seed protocol.

## Final-test firewall

`test` and `normal_test` are canonical final-test material. Development trainers/diagnostics/evaluators refuse them. Canonical final evaluation is permitted only through `scripts/run_final_bundle_v21.py` after all models, thresholds, metrics and statistical decisions are frozen. Until then, the final test remains **CLOSED**.
