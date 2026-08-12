# OASIS conditional upgrade: Phase A-B result

## Decision

**Student architecture: promising. OASIS contribution: NO-GO.**

The multi-scale RGB-only student improved the validation result, but the conditional OASIS critic failed the RGB/mask shortcut gate. The critic was therefore not connected to the student and no OASIS improvement claim is allowed.

## Protocol

- Resolution: 128 x 128
- Device: CPU
- Seed: 1337
- Epochs: 12
- Threshold: validation only
- `test_debug`: not used for selection or this report
- Deployment contract: RGB-only student

## Student control

The new multi-scale student was compared with the previous augmented B0 control:

| Model | Val Dice | Val IoU | Normal FP px/image |
|---|---:|---:|---:|
| Previous B0 control | 0.50250 | 0.33556 | 1.20 |
| Multi-scale RGB-only student | **0.58919** | **0.41762** | **0.00** |

This is evidence for the multi-scale student on the current tiny validation pilot, not evidence for OASIS. It still requires three seeds and a valid source-disjoint test.

## Conditional critic qualification

The first critic configuration collapsed invalid-mask recognition. Increasing the class-2 weight from 2 to 40 improved the semantic metrics:

| Metric | Balanced critic |
|---|---:|
| Real pixel accuracy | 97.97% |
| Crack recall | 81.69% |
| Invalid-mask recall | 97.77% |
| RGB-flip invalid detection | **2.79%** |
| Mask-flip invalid detection | **31.61%** |

The critic passes the raw crack/invalid recall thresholds narrowly, but fails the causal shortcut tests. It does not reliably use RGB-mask consistency; it mainly reacts to mask patterns/artifacts. Therefore `lambda_oasis` remains zero and Phase C is blocked.

## Code completed

- Added `MultiScaleLightweightSegmenter` with 1/1, 1/2, 1/4 and 1/8 features.
- Added `ConditionalOASISCritic(RGB, mask)`.
- Added corrupted-mask batch protocol.
- Added orientation-selected AOSK consistency loss without circular shifts.
- Added `train_upgrade.py` with `multiscale`, `critic` and gated `full` modes.
- Added shape/gradient tests.
- Recovered CPU PyTorch runtime for execution.

## Gate conclusion

```text
Multi-scale RGB student: PASS as an architecture candidate.
Conditional OASIS critic: FAIL shortcut/causal qualification.
OASIS-connected student: NOT TRAINED.
Test opening: BLOCKED.
Paper claim: BLOCKED.
```

The next research task is to redesign the critic qualification task so that RGB and mask are spatially aligned and balanced, then repeat the shortcut tests. Do not connect the current critic to the student merely because its raw invalid-mask recall is high.

## Deployment cost audit

The multi-scale student control has:

- 211,073 trainable parameters (0.211M);
- approximately 19.24 GFLOPs at 128 x 128 using Conv2d multiply-add counting;
- mean CPU latency 9.73 ms and p95 latency 17.51 ms in the current runtime;
- checkpoint keys limited to `student`, configuration and inference metadata;
- deployment contract: `RGB -> crack logits only`.

These cost figures are engineering diagnostics on the current CPU runtime, not a hardware benchmark claim.

## Relational critic retry

To address the shortcut failure, the critic was upgraded with separate RGB/mask encoders, multiplicative feature interaction and explicit RGB/mask misalignment negatives.

| Variant | Crack recall | Invalid-mask recall | RGB-flip invalid recall | Mask-flip invalid recall |
|---|---:|---:|---:|---:|
| Relational v1 | 82.08% | 96.01% | 97.41% | 52.06% |
| Relational v2 | 79.24% | 98.97% | 98.42% | 79.65% |
| Relational v3 | 74.96% | 99.08% | 98.34% | 75.51% |

The trade-off is not acceptable: improving mask-mismatch detection reduces real crack recall below the 80% gate. No relational variant simultaneously passes semantic recall and shortcut sensitivity. Phase C (multi-scale + OASIS) and Phase D (AOSK integration) therefore remain blocked.

```text
OASIS conditional critic: FAIL quality gate after targeted redesign.
Multi-scale student: remains the only validated improvement.
```
