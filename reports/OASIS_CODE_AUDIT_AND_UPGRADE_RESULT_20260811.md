# Full OASIS code audit and upgrade result

Date: 2026-08-11

## Executive conclusion

The code is runnable and deployment isolation passes, but OASIS does not show a
stable segmentation gain. Increasing critic quality and adding direct
corrupted-mask ranking improved critic diagnostics but did not recover the old
MobileNet gain. The old high result cannot be restored using the old unpaired
RNG protocol.

## Main audit findings

- Canonical entrypoint: `train_oasis_rc.py`; older train entrypoints implement
  incomparable experiments.
- Student-only checkpoints do not require critic/generator/AOSK.
- Test is not loaded by the trainer; nearest-neighbor mask resize is used.
- Student RNG is reset after critic construction for paired control/connected
  initialization.
- Strict source-disjoint audit is opt-in and must be enforced externally.
- Current AOSK is not the originally specified PCA/skeleton/width-aware kernel.
- The evaluator originally lacked thin-crack, skeleton, fragmentation and
  bridge metrics.
- The historical pair head was weakly calibrated.

## Evaluated variants

| Variant | Validation Dice | Decision |
|---|---:|---|
| OASIS-RC v1 | 0.38141 | no gain over control 0.38760 |
| v2, lambda 0.003 | 0.38767 | numerical tie |
| v2, lambda 0.010 | 0.38114 | reject |
| v2, lambda 0.030 | 0.38474 | reject |
| pair-weight critic + v2 | 0.38051 | reject |

Increasing pair-consistency weight from `0.25` to `1.0` changed RGB mismatch
drop from `0.0517` to `0.2413` and mask mismatch drop from `0.0522` to `0.2459`,
but student Dice still fell. Better critic diagnostics did not prove useful
student supervision.

## Decision

```text
Code correctness: PASS with legacy-entrypoint and audit caveats
OASIS critic quality: improved by pair-weight variant
OASIS student gain: NOT ESTABLISHED
Old high result: not accepted as reproducible evidence
Recommended continuation: student-aware OASIS-RC v3
Deployment/Q1 claim: BLOCKED
```
