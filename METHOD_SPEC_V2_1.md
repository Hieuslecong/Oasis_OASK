# OASIS-RC-v2.1 Method Specification

Status: **scientific revision / pre-confirmatory**

This document is the source of truth for OASIS-RC-v2.1. The historical/reconstructed v2.0.4 formulation remains immutable evidence and must not be described as bit-for-bit historical OASIS-RC-v2.

## 1. Scientific question

Can training-only structured relational and orientation-aware supervision improve lightweight RGB-only crack segmentation, including topology-sensitive and false-positive outcomes, without changing the inference graph?

## 2. Inference contract

Canonical deployment is unchanged:

`RGB -> lightweight student -> crack logits/mask`

The relation critic and AOSK are training-only. They must not be serialized into deployment checkpoints or invoked at inference.

## 3. Canonical relation energy

OASIS-RC-v2.1 uses a **dedicated scalar relation-energy head** in the training-only critic. The classification/mismatch/pair heads remain representation supervision, but they no longer define the scientific energy implicitly.

For critic output `out`:

`E(out) = out["energy"]`.

Lower energy means a more compatible RGB-mask relation.

For image `I`, ground truth `G`, student soft prediction `P`, and structured corruption `C`:

- `E_G = E(I,G)`
- `E_P = E(I,P)`
- `E_C = E(I,C)`

`E_G` and `E_C` are detached during student optimization. Gradients from relational supervision are allowed to update the student only.

## 4. Critic training objective

The critic has two roles that are trained and qualified separately.

### 4.1 Representation/classification supervision

The reconstructed representation contract is retained:

`L_repr = 0.5 * (L_clean + L_corrupt) + lambda_rgb * L_rgb_pair + lambda_normal * L_normal_donor`.

This includes semantic valid-background/valid-crack/invalid supervision, mismatch supervision, pair validity, RGB-shuffle pair supervision, and optional true-normal donor supervision.

### 4.2 Dedicated energy calibration

Ground-truth energy is anchored near zero:

`L_E_anchor = SmoothL1(E_G, 0)`.

The corruption endpoint is required to be worse than GT:

`L_E_endpoint = relu(E_G - E_C + margin_endpoint)`.

For a continuous corruption trajectory

`M_t = (1-t)G + tC`, `t in [0,1]`,

adjacent energies are ordered by

`L_E_path = mean_i relu(E(M_ti) - E(M_tj) + margin_path * (tj-ti))`, for `tj > ti`.

Canonical critic loss:

`L_critic = L_repr + lambda_endpoint * (anchor_weight * L_E_anchor + L_E_endpoint) + lambda_path * L_E_path`.

The dedicated energy head exists because classifier probabilities are not guaranteed to induce the required lower-is-better GT-to-corruption ordering.

## 5. Canonical v2.1 student relational objective

The v2.0.4 midpoint-ranking objective is retired for v2.1. v2.1 uses a GT anchor plus one-sided corruption rejection:

`L_anchor = SmoothL1(E_P, E_G)`

`L_reject = relu(E_P - E_C + margin_student)`

`L_relation = L_anchor + corrupted_rank_weight * L_reject`

The background false-positive term remains:

`L_FP = sum((1-G) * P * sigmoid(mismatch_pred)) / max(sum(1-G), 1)`

Canonical auxiliary loss:

`L_RC = L_relation + fp_weight * L_FP`

Canonical student loss:

`L_student = L_seg + lambda_RC * L_RC + lambda_AOSK * L_AOSK`.

### Required invariants

1. If `E_P == E_G` and `E_G + margin_student <= E_C`, the relational ranking component must be zero and must not repel the prediction from GT.
2. If `E_P > E_C - margin_student`, gradient descent must decrease `E_P`.
3. `E_G` and `E_C` must be detached for student updates.
4. Critic parameters must receive no gradient during student updates.
5. Connected arms are forbidden unless both critic representation qualification and relation-energy usability qualification pass.
6. Critic checkpoints without the dedicated energy-head contract are incompatible with v2.1-dev1.

## 6. Critic calibration and qualification requirement

A binary valid/invalid classifier is not sufficient evidence that the critic provides a useful student loss. Before connected real-data training, v2.1 must qualify the energy landscape on held-out validation trajectories.

At minimum the report must include:

- positive GT-to-corruption energy-gap fraction;
- median/mean `E_C - E_G`;
- continuous-path monotonicity/order;
- soft student-prediction energy distribution;
- relational gradient norm;
- ratio of relational to segmentation gradient norms;
- cosine similarity between relational and segmentation gradients;
- saturation/finiteness diagnostics.

Development thresholds are frozen before confirmatory runs. Canonical test data must not be used to set them.

## 7. AOSK claim boundary

Canonical AOSK is an image-guided orientation-aware local consistency regularizer. It is **not** itself a topology loss. Topology-preservation claims require independent topology metrics and comparison against a strong topology-aware baseline such as clDice.

Flat/low-texture regions must use an isotropic fallback or disable orientation preference when local orientation evidence is insufficient.

## 8. Experimental claims

Minimum paired arms:

- S0/B0: segmentation control
- B1: segmentation + clDice
- B2: segmentation + conventional pair-adversarial supervision
- S1: segmentation + RC-v2.1
- S2: segmentation + AOSK
- S3: segmentation + RC-v2.1 + AOSK

The incremental RC claim requires `S1 > S0` and, for the combined model, `S3 > S2` under preregistered metrics. Tiny or inconsistent effects must be reported as null/negative rather than rescued by post-hoc tuning.

## 9. Development versus confirmatory evidence

Any seed inspected before changing method or hyperparameters is development evidence. Once development evidence is used to tune the method, it must not be counted as an independent confirmatory seed.

No canonical final-test access is permitted before the method, effective configuration, evaluator, metrics, checkpoints, thresholds, and multi-arm evaluation bundle are frozen and hashed.

## 10. Version identity

- Method: `OASIS-RC-v2.1`
- Implementation: `2.1.0-dev1`
- Checkpoint schema: `5`
- Experiment family: `oasis-rc-v2.1-gt-anchored-relational-energy-head`

A future confirmatory release must replace the development implementation version only after the relation-energy usability gate, real-data training pipeline, evaluation protocol, and final-test bundle are complete.
