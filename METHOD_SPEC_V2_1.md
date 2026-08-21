# OASIS-RC-v2.1 Method Specification

Status: **scientific revision / pre-confirmatory**

This document is the source of truth for OASIS-RC-v2.1. The historical/reconstructed v2.0.4 formulation remains immutable evidence and must not be described as bit-for-bit historical OASIS-RC-v2.

## 1. Scientific question

Can training-only structured relational and orientation-aware supervision improve lightweight RGB-only crack segmentation, including topology-sensitive and false-positive outcomes, without changing the inference graph?

## 2. Inference contract

Canonical deployment is unchanged:

`RGB -> lightweight student -> crack logits/mask`

The relation critic and AOSK are training-only. They must not be serialized into deployment checkpoints or invoked at inference.

## 3. Relation energy

For critic output `out`:

`E(out) = mean(sigmoid(mismatch)) + pair_weight * (1 - sigmoid(pair))`.

Lower energy means a more compatible RGB-mask relation.

For image `I`, ground truth `G`, student soft prediction `P`, and structured corruption `C`:

- `E_G = E(I,G)`
- `E_P = E(I,P)`
- `E_C = E(I,C)`

`E_G` and `E_C` are detached during student optimization. Gradients from relational supervision are allowed to update the student only.

## 4. Canonical v2.1 student relational objective

The v2.0.4 midpoint-ranking objective is retired for v2.1. v2.1 uses a GT anchor plus one-sided corruption rejection:

`L_anchor = SmoothL1(E_P, E_G)`

`L_reject = relu(E_P - E_C + margin)`

`L_relation = L_anchor + corrupted_rank_weight * L_reject`

The background false-positive term remains:

`L_FP = sum((1-G) * P * sigmoid(mismatch_pred)) / max(sum(1-G), 1)`

Canonical auxiliary loss:

`L_RC = L_relation + fp_weight * L_FP`

Canonical student loss:

`L_student = L_seg + lambda_RC * L_RC + lambda_AOSK * L_AOSK`.

### Required invariants

1. If `E_P == E_G` and `E_G + margin <= E_C`, the relational ranking component must be zero (up to numerical precision) and must not repel the prediction from GT.
2. If `E_P > E_C - margin`, gradient descent must decrease `E_P`.
3. `E_G` and `E_C` must be detached for student updates.
4. Critic parameters must receive no gradient during student updates.
5. Connected arms are forbidden unless both critic representation qualification and relation-energy usability qualification pass.

## 5. Critic calibration requirement

A binary valid/invalid classifier is not sufficient evidence that the critic provides a useful student loss. Before confirmatory training, v2.1 must qualify the energy landscape on continuous masks.

For structured corruption `C`, define a corruption trajectory:

`M_t = (1-t)G + tC`, where `t in [0,1]`.

Qualification must evaluate monotonicity/order along held-out trajectories and on real soft student predictions. At minimum the report must include:

- positive GT-to-corruption energy-gap fraction;
- median/mean `E_C - E_G`;
- continuous-path monotonicity/order;
- soft student-prediction energy distribution;
- relational gradient norm;
- ratio of relational to segmentation gradient norms;
- cosine similarity between relational and segmentation gradients;
- saturation/finiteness diagnostics.

Thresholds used as PASS/FAIL criteria must be fixed using development data before confirmatory runs.

## 6. AOSK claim boundary

Canonical AOSK is an image-guided orientation-aware local consistency regularizer. It is **not** itself a topology loss. Topology-preservation claims require independent topology metrics and comparison against a strong topology-aware baseline such as clDice.

Flat/low-texture regions must not receive an arbitrary directional preference; the implementation must use an isotropic fallback or disable orientation preference when local orientation evidence is insufficient.

## 7. Experimental claims

Minimum paired arms:

- S0: segmentation control
- S1: segmentation + RC-v2.1
- S2: segmentation + AOSK
- S3: segmentation + RC-v2.1 + AOSK

Strong baselines should additionally include clDice and a conventional adversarial segmentation critic under matched student/data/update budgets.

The incremental RC claim requires `S1 > S0` and, for the combined model, `S3 > S2` under preregistered metrics. Tiny or inconsistent effects must be reported as null/negative rather than rescued by post-hoc tuning.

## 8. Development versus confirmatory evidence

Any seed inspected before changing method or hyperparameters is development evidence. Once development evidence is used to tune the method, it must not be counted as an independent confirmatory seed.

No canonical final-test access is permitted before the method, effective configuration, evaluator, metrics, checkpoints, thresholds, and multi-arm evaluation bundle are frozen and hashed.

## 9. Version identity

- Method: `OASIS-RC-v2.1`
- Implementation: `2.1.0-dev`
- Experiment family: `oasis-rc-v2.1-gt-anchored-relational-energy`

A future confirmatory release must replace the `-dev` implementation version only after the relation-energy usability gate, evaluation protocol, and final-test bundle are complete.
