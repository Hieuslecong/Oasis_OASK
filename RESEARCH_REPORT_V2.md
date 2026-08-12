# OASIS-RC-v2: rationale, architecture, protocol and evidence

Date: 2026-08-12

## Executive conclusion

OASIS-RC-v2 is a training-only relational critic extension for RGB-only
lightweight crack segmentation. It was built because OASIS-RC v1 could
distinguish valid and synthetic corrupted masks, but its global
GT-vs-prediction signal did not transfer into a stable student improvement.
V2 adds an explicit online corrupted-mask ranking reference.

The hypothesis is reasonable, but current evidence is negative. The strongest
one-seed v2 result is numerically tied with control, and the later four-arm
critic failed its preregistered RGB/mask pair-drop gates. V2 is therefore not a
validated Q1 contribution.

## 1. Research lineage

### 1.1 OASIS principle

OASIS replaces a generic real/fake discriminator with pixel-level semantic
adversarial supervision. Its discriminator predicts real semantic classes plus
a fake class. OASIS-RC retains the semantic-discriminator principle but changes
the task: it evaluates whether an RGB image and a candidate crack mask are
spatially consistent. It is not used to synthesize an independent dataset.

### 1.2 Crack segmentation requirements

Crack masks are sparse, thin and topological. Deep hierarchical crack
segmentation work motivates preserving multi-scale structure and evaluating
more than pixel accuracy. A background-dominant critic can appear accurate
while missing almost every crack pixel, so crack recall and structural metrics
are mandatory.

### 1.3 Lightweight deployment

MobileNetV3 motivates the RGB-only lightweight student. The critic is removed
after training, preserving deployment cost. This separation is central to the
method: OASIS-RC-v2 may increase training cost but must not increase inference
parameters or latency.

### 1.4 Relational and dense supervision

Dense contrastive learning and mask-embedded discriminator research motivate
spatially aligned image-mask comparisons instead of image-level realism alone.
OASIS-RC uses separate image/mask encoders and multiplicative/difference fusion
to expose local agreement and disagreement.

## 2. Why v1 was insufficient

V1 uses:

```text
L_student_v1 = BCE + Dice + lambda * (L_rank(pred, GT) + L_FP)
```

The relational energy is dominated by global averages. Synthetic corruption
quality can be high while the critic remains poorly aligned with errors the
student actually produces. The three-seed MobileNet mean improvement was only
about `+0.000054` Dice, effectively zero. This motivated a stronger relative
reference rather than blind lambda escalation.

## 3. V2 hypothesis

For each training image, construct three in-batch relations:

```text
R_gt        = (RGB, GT mask)
R_pred      = (RGB, student soft mask)
R_corrupted = (RGB, online corrupted GT mask)
```

The critic relation energy is:

```text
E(R) = mean(sigmoid(mismatch_logits))
     + alpha_pair * (1 - sigmoid(pair_valid_logit))
```

V2 optimizes:

```text
L_rank_gt = softplus(E_pred - E_gt + margin)
L_rank_corrupt = softplus(E_pred - E_corrupted + margin)
L_FP = sum((1-GT) * P_student * Q_mismatch) / sum(1-GT)

L_student_v2 = BCE + Dice
             + lambda_oasis * (
                 L_rank_gt
               + w_corrupt * L_rank_corrupt
               + L_FP)
```

`E_gt` and `E_corrupted` are detached. Critic parameters are frozen during the
student update. Gradients pass through `R_pred` only.

## 4. Architecture

### 4.1 Student

The principal student is MobileNetV3-Small-style segmentation with a compact
decoder and one-channel crack logits. Additional smoke controls include
DS-UNet-Lite, Fast-SCNN-Lite and BiSeNet-Tiny. All are RGB-only at inference.

### 4.2 Critic

The critic has parallel image and mask encoders at three scales. At each scale:

```text
F_rel = concat(F_I, F_M, F_I * F_M, abs(F_I - F_M))
```

A decoder produces crack-consistency and invalidity maps. Semantic logits are
composed as:

```text
z_background = -z_crack - z_invalid
z_crack      =  z_crack - z_invalid
z_invalid    =  z_invalid
```

A global pair head predicts RGB-mask validity. Outputs are semantic
`B x 3 x H x W`, mismatch `B x 1 x H x W`, and pair `B x 1`.

### 4.3 Online corruptions

The implementation supports zero-padded shifts, dilation, erosion, fragment
removal, donor masks, texture blobs, RGB flip, mask flip and normal-image plus
crack-mask pairing. Corruptions are generated in the current batch and never
stored as a dataset.

## 5. Difference table

| Property | OASIS-RC | OASIS-RC-v2 |
|---|---|---|
| Deployment | RGB-only student | RGB-only student |
| Critic input | RGB + soft/GT mask | same |
| Critic outputs | semantic, mismatch, pair | same |
| Student relation references | GT and prediction | GT, prediction and corrupted GT |
| Ranking | prediction vs GT | prediction vs GT plus prediction vs corrupted |
| FP loss | GT background only | GT background only |
| Pair weight | fixed historical value | exposed ablation parameter |
| Main risk | weak/global transfer | synthetic corruption shortcut |

## 6. Historical evidence

### 6.1 One-seed audit

| Variant | Validation Dice | Decision |
|---|---:|---|
| Control | 0.38760 | reference |
| OASIS-RC v1 | 0.38141 | worse |
| V2 lambda 0.003 | 0.38767 | numerical tie |
| V2 lambda 0.010 | 0.38114 | reject |
| V2 lambda 0.030 | 0.38474 | reject |
| Pair-weight critic + v2 | 0.38051 | reject |

Increasing critic pair weight improved mismatch sensitivity but reduced student
Dice. Critic diagnostics alone therefore do not prove useful supervision.

### 6.2 Four-arm smoke

The later relation-aware critic obtained valid crack recall `0.83975` and
invalid recall `0.93291`, but RGB pair drop `0.01576` and mask pair drop
`0.01545`, below the `0.05` gate. OASIS-connected arms were blocked. AOSK had a
positive validation signal, but that is not evidence for an OASIS contribution.

## 7. Reproducibility status

The package includes runnable reconstructed v2 code, tests, exact scripts,
configs, historical artifacts and reports. It does not include raw image data.
Historical manifests contain environment-specific absolute paths and must be
remapped. The active debug manifest also leaks `BCL` and `normal` sources across
splits, so it cannot certify a paper result.

The exact historical v2 source snapshot is missing. New results must be called
reconstructed-v2 results and reported separately.

## 8. Required evaluation

Run same-backbone controls, three seeds, validation-only thresholds and a
source-disjoint locked test. Report precision, recall, Dice/IoU, clDice,
skeleton precision/recall, thin recall, fragmentation, false bridges, normal
FPR, false-positive pixels/components per image, maximum FP component,
parameters, FLOPs, latency and memory.

Reject v2 if the critic fails semantic/shortcut gates, if any seed is blocked,
or if paired gains are unstable or accompanied by structural/normal-FP harm.

## 9. Novelty and Q1 positioning

Potential novelty is not “using OASIS for crack segmentation” alone. A defensible
claim would require evidence that a training-only conditional semantic
relational critic, with explicit corrupted-mask ordering, improves multiple
lightweight RGB-only students while preserving topology and edge-device cost.

Current evidence does not establish that. The honest contribution today is a
reproducible negative study and a falsifiable design framework. Q1 readiness is
blocked by source leakage, missing exact historical v2 source, incomplete
structural evaluation and absence of a stable OASIS gain.

## 10. Decision

```text
Reconstructed v2 code: runnable candidate
Exact historical v2 reproduction: unavailable
Critic qualification: unstable/failed in latest smoke
Student gain over control: not established
Q1/TIM/ECCV positive claim: blocked
```
