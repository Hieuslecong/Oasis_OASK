# Three-seed OASIS-RC / AOSK ablation status

Student: MobileNetV3-Small RGB-only, 128x128, BCE + Dice, AdamW, 12 epochs.
Seeds: 1337, 2027, 31415. Checkpoints and thresholds were selected on
validation only.

## Critic qualification

| Seed | Valid crack recall | Invalid recall | Pair accuracy | Decision |
|---:|---:|---:|---:|---|
| 1337 | 0.9801 | 0.9180 | 0.9000 | pass |
| 2027 | 0.9788 | 0.9082 | 0.8333 | pass |
| 31415 | 0.7207 | 0.9072 | 0.7667 | fail |

Connected OASIS arms were blocked at seed 31415. No valid three-seed OASIS
claim exists.

## Validation

| Variant | Mean Dice | Mean IoU | Status |
|---|---:|---:|---|
| Control | 0.3650±0.0171 | 0.2233±0.0128 | baseline |
| AOSK skeleton-aware | 0.3944±0.0119 | 0.2457±0.0093 | validation signal only |
| OASIS-RC | incomplete | incomplete | seed 31415 blocked |
| OASIS-RC+AOSK | incomplete | incomplete | seed 31415 blocked |

Both available manifests fail strict source-disjoint audit because `BCL` and
`normal` occur across partitions.

Required wording:

> OASIS-RC branch is not qualified for deployment or a positive paper claim.
