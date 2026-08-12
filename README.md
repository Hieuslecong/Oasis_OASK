# OASIS-RC-v2 reproducibility package

This folder separates OASIS-RC-v2 from the original OASIS-RC package.

## Provenance warning

The exact historical v2 source used for the previously reported v2 experiments
was not present in the current workspace snapshot. Only checkpoints/result
folders and audit reports survived. Therefore:

- `src/oasis_cycle_aosk/train_oasis_rc_v2.py` and `losses_v2.py` are a
  reconstructed canonical implementation based on the audited v2 definition;
- historical numbers are preserved in `reports/` and `artifacts/`;
- this package can rerun the declared v2 hypothesis, but it cannot promise
  bit-for-bit reproduction of the historical v2 numbers.

Calling this exact historical reproduction would be scientifically false.

## Difference from OASIS-RC v1

OASIS-RC v1 uses a GT-vs-student relation and background false-positive loss.
V2 adds a third reference generated online from the same GT mask:

```text
(RGB, GT mask)         -> valid reference
(RGB, student mask)    -> differentiable candidate
(RGB, corrupted mask)  -> explicit invalid reference
```

The v2 objective is:

```text
L_v2 = L_BCE + L_Dice
     + lambda_oasis * (
         L_rank(prediction, GT)
       + w_corrupt * L_rank(prediction, corrupted mask)
       + L_FP_on_GT_background
       )
```

Corrupted masks are generated online only. They are never saved as a dataset.
Inference remains strictly RGB -> lightweight student -> crack mask.

## Historical result status

The one-seed audit found:

```text
control Dice              0.38760
v1 Dice                   0.38141
v2 lambda=0.003 Dice      0.38767
v2 lambda=0.010 Dice      0.38114
v2 lambda=0.030 Dice      0.38474
pair-weight+v2 Dice       0.38051
```

The best v2 value is a numerical tie with control, not a credible gain. In a
later four-arm smoke, the v2 critic passed crack/invalid recall but failed both
RGB and mask pair-drop gates; connected v2 arms were correctly blocked.

## Folder map

```text
src/oasis_cycle_aosk/       v1 snapshot plus reconstructed v2 entrypoint/loss
configs/                    three seed configs
tests/                      shared tests and v2 gradient/contract tests
scripts/                    v2 smoke, full and evaluation runners
manifests/                  diagnostic manifests; raw images are not embedded
artifacts/                  surviving historical runs/checkpoints
reports/                    v2 audit, four-arm smoke and prior reports
REPRODUCE_TRAIN_TEST_PROMPT.md
RESEARCH_REPORT_V2.md
HISTORICAL_SOURCE_GAP.md
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e .
pytest -q
```

Remap the historical manifest paths as described in
`REPRODUCE_TRAIN_TEST_PROMPT.md`, then run:

```bash
scripts/run_smoke.sh /absolute/path/to/manifest.jsonl mobilenetv3
scripts/run_lightweight_smoke.sh /absolute/path/to/manifest.jsonl
```

The v2 branch is currently rejected as a positive paper claim. Its value is a
well-defined negative experiment and a basis for a future student-aware v3.
