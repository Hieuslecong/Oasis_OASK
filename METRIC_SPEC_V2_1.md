# OASIS-RC-v2.1 Metric Specification

Status: **development freeze candidate / confirmatory input**

This file defines the metric semantics that must be hashed into the immutable final-test bundle. Canonical final evaluation must not change these definitions after the final bundle is opened.

## 1. Populations

Crack overlap and topology metrics are computed only on rows whose ground-truth mask contains at least one positive pixel. True-negative rows are excluded from Dice/IoU/clDice aggregation and are evaluated only with false-positive metrics.

Canonical final populations are:

- `test`: crack benchmark population.
- `normal_test`: independent true-normal robustness population.

Both are final-test material and must be opened under one immutable bundle/ledger event.

## 2. Pixel metrics on crack-positive rows

Let TP, FP and FN be aggregated over crack-positive rows only.

- Precision = `TP / (TP + FP)`.
- Recall = `TP / (TP + FN)`.
- Dice/F1 = `2 TP / (2 TP + FP + FN)`.
- IoU = `TP / (TP + FP + FN)`.

`macro_dice` and `macro_iou` are the arithmetic means of per-image crack-positive Dice and IoU respectively.

## 3. Structural metrics on crack-positive rows

- `cldice`: centerline Dice computed from the frozen soft-centerline implementation used by `evaluate_v21.py`.
- `skeleton_precision`: predicted-centerline overlap with the ground-truth foreground.
- `skeleton_recall`: ground-truth-centerline overlap with the predicted foreground.
- `mean_component_excess`: mean of `max(0, predicted_components - target_components)` using 8-connected components.

Lower `mean_component_excess` is better. The AOSK loss itself is not a topology metric and must not be used as evidence of topology preservation.

## 4. True-normal robustness metrics

For rows with an empty ground-truth mask:

- `normal_fp_pixels_mean`: mean number of predicted foreground pixels per normal image.
- `normal_fp_components_mean`: mean number of 8-connected predicted foreground components per normal image.
- `normal_any_fp_rate`: fraction of normal images with at least one predicted foreground pixel.

All three are lower-is-better. Dice, IoU and clDice are undefined for true-negative rows and must not be imputed as zero or one for aggregate crack performance.

## 5. Threshold policy

Each checkpoint uses its frozen `threshold_validation`, selected only from the development validation population. Canonical final evaluation must use exactly that checkpoint threshold; no final-test threshold search, recalibration or normal-test tie-break is permitted.

## 6. Confirmatory comparisons

Preregistered contrasts are:

- `B1 - B0`
- `B2 - B0`
- `S1 - B0`
- `S2 - B0`
- `S3 - S2`

Positive reported delta always means the treatment is better. For lower-is-better metrics, the analysis therefore reports `base - treatment`; otherwise it reports `treatment - base`.

The sampling unit for confirmatory method uncertainty is the training seed. The canonical seed set is `2027, 31415, 42421, 51511, 62617`. Every preregistered metric must be finite and available for every paired seed; missing values fail closed rather than reducing the number of seed pairs post hoc.

Report paired seed mean delta, standard deviation, median delta, 95% seed-bootstrap CI for the mean, direction consistency, Cohen's dz, and the exact two-sided paired sign-flip p-value. Exact p-values are secondary evidence with five seeds. Holm multiplicity correction is applied within each preregistered metric family.

## 7. Frozen implementation references

The immutable final bundle must bind hashes for:

- this metric specification;
- `METHOD_SPEC_V2_1.md`;
- `protocols/real_data_v21.json`;
- the final evaluator implementation;
- every student checkpoint and its frozen validation threshold;
- the full Gate0 certificate and final dataset bytes;
- the exact Git commit used by all bundled student checkpoints.
