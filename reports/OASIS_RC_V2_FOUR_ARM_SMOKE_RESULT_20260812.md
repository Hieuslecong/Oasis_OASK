# OASIS-RC v2 four-arm smoke result

Status: diagnostic only; active manifest is not source-disjoint paper evidence.

## Critic gate

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| Valid crack recall | 0.83975 | >=0.80 | PASS |
| Invalid recall | 0.93291 | >=0.90 | PASS |
| RGB pair drop | 0.01576 | >=0.05 | FAIL |
| Mask pair drop | 0.01545 | >=0.05 | FAIL |

S1 and S3 were blocked before connected student training.

## Validation, three seeds

| Variant | Dice/F1 | IoU | clDice | Skeleton recall | Thin recall | Fragmentation | False bridges |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 Control | 0.35132±0.04757 | 0.21378±0.03562 | 0.47334±0.04970 | 0.49649±0.07450 | 0.47734±0.08301 | 7.67±0.58 | 3.67±0.58 |
| S1 OASIS-RC v2 | blocked | blocked | blocked | blocked | blocked | blocked | blocked |
| S2 AOSK | 0.37169±0.03125 | 0.22857±0.02371 | 0.49316±0.02707 | 0.48790±0.04876 | 0.47526±0.05245 | 6.67±1.53 | 3.33±0.58 |
| S3 v2+AOSK | blocked | blocked | blocked | blocked | blocked | blocked | blocked |

Paired S2-S0 Dice delta was `+0.02038`, but this is an AOSK signal, not an
OASIS contribution.

## Decision

OASIS-RC v2 and the combined arm were rejected/blocked under the preregistered
critic gate. No Q1/TIM/ECCV claim is supported by this smoke.
