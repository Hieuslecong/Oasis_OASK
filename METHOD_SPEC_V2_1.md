# OASIS-RC-v2.1 Method Specification

Status: **scientific revision / development / pre-confirmatory**

This document is the source of truth for OASIS-RC-v2.1. The historical/reconstructed v2.0.4 formulation remains immutable evidence and must not be described as bit-for-bit historical OASIS-RC-v2.

## 1. Scientific question

Can training-only structured relational and orientation-aware supervision improve lightweight RGB-only crack segmentation, including topology-sensitive and false-positive outcomes, without changing the inference graph?

## 2. Inference contract

Canonical deployment is unchanged:

`RGB -> lightweight student -> crack logits/mask`

The relation critic and AOSK are training-only. They must not be serialized into deployment checkpoints or invoked at inference.

## 3. Canonical relation energy

OASIS-RC-v2.1 uses a **dedicated scalar relation-energy head** in the training-only critic. The classification/mismatch/pair heads remain representation supervision, but they do not define the scientific energy implicitly.

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

`L_anchor` is intentionally an anchor, not a monotonic preference for arbitrarily low energy: if `E_P < E_G`, the symmetric SmoothL1 term pulls the prediction back toward the GT relation energy.

### Required invariants

1. If `E_P == E_G` and `E_G + margin_student <= E_C`, the relational ranking component must be zero and must not repel the prediction from GT.
2. If `E_P > E_C - margin_student`, gradient descent must decrease `E_P`.
3. `E_G` and `E_C` must be detached for student updates.
4. Critic parameters must receive no gradient during student updates.
5. Connected arms are forbidden unless both critic representation qualification and relation-energy usability qualification pass.
6. Critic consumers must declare the complete v2.1 scientific compatibility contract; historical optimizer settings are provenance rather than student-run compatibility requirements.
7. Every v2.1 critic-consuming launch must re-run qualification from the currently loaded critic weights. A stored `qualification_v21.pass` is provenance, not sole authorization.
8. Legacy v2.0.4 entrypoints must not be capable of creating v2.1 connected-arm evidence.

## 6. Critic calibration and qualification

A binary valid/invalid classifier is not sufficient evidence that the critic provides a useful student loss. Before connected real-data training, v2.1 must qualify the energy landscape on held-out validation trajectories.

At minimum the report must include:

- positive GT-to-corruption energy-gap fraction;
- the **true global** median and mean `E_C - E_G` across validation samples;
- continuous-path monotonicity/order;
- soft student-prediction energy distribution;
- relational gradient norm;
- raw relational/segmentation gradient ratio;
- `lambda_RC * ramp * relational/segmentation gradient ratio`;
- cosine similarity between relational and segmentation gradients;
- saturation/finiteness diagnostics.

Development thresholds are frozen before confirmatory runs. Canonical test data must not be used to set them.

For N25, critic qualification must use `normal_val`; falling back to the normal training rows is not admissible.

## 7. Auxiliary-weight calibration

Parser defaults are development starting points, not confirmatory evidence. Before confirmatory seeds, each auxiliary coefficient must be calibrated using the development seed only under one declared rule and then frozen.

The repository must log both raw and lambda-weighted gradient contribution. Gradient magnitude is a sanity/strength diagnostic, not a requirement that heterogeneous objectives have identical norms. Calibration must consider at least effective gradient contribution, validation response, instability/non-finiteness, and degradation of the primary segmentation objective. No coefficient may be changed after confirmatory runs begin.

## 8. AOSK v2 claim boundary

Canonical dev2 AOSK is `structure-tensor-steered-v2`:

- local RGB gradients form the 2x2 structure tensor using `Jxx`, `Jyy`, and the cross-term `Jxy`;
- the principal gradient-normal orientation is converted to a tangent direction;
- logits are sampled at `+/- tangent` using differentiable bilinear sampling;
- low-coherence regions smoothly fall back to isotropic local consistency.

This supports arbitrary local angles rather than only horizontal/vertical preference. AOSK remains **a training-only local-consistency regularizer, not a topology loss**. Topology/continuity/junction claims require independent topology metrics and comparison against a topology-aware baseline such as clDice.

## 9. Experimental arms and claim semantics

Canonical arm identifiers are:

- B0: BCE + Dice segmentation control
- B1: B0 + clDice
- B2: B0 + **frozen pretrained pair-critic BCE**; this is an ablation and must not be called conventional jointly-trained adversarial training
- S1: B0 + OASIS-RC-v2.1
- S2: B0 + AOSK structure-tensor-v2
- S3: B0 + OASIS-RC-v2.1 + AOSK structure-tensor-v2

If a conventional adversarial baseline is required for a paper claim, it must be implemented as a separate discriminator with its own jointly/alternately optimized parameters; B2 is not that baseline.

Primary preregistered contrasts are:

- `B1 - B0`
- `B2 - B0`
- `S1 - B0`
- `S2 - B0`
- `S3 - S2`

Tiny or inconsistent effects must be reported as null/negative rather than rescued by post-hoc tuning.

## 10. Data protocols and final-test firewall

N0 excludes external true-normal supervision. Certified native-empty rows belonging to the crack dataset may remain internal true negatives.

N25 uses a fixed 25% normal batch-composition protocol with lineage-disjoint `normal_train`, `normal_val`, and `normal_test`. A `train_and_aux_val` Gate0 certificate is strictly stronger than `train`; N0 accepts `none` only.

Both `test` and `normal_test` are canonical final-test material. Development trainers, diagnostics, and evaluators must refuse either split. The immutable final-bundle runner is the sole sanctioned route and must evaluate both splits under one content-addressed single-open ledger marker.

## 11. Evaluation and statistics

Accuracy and topology metrics are computed on crack-positive rows. True-negative rows are evaluated with dedicated false-positive metrics; crack-overlap metrics must never be averaged over true-negative rows.

Confirmatory uncertainty uses **training seed as the sampling unit**. Image rows within one trained checkpoint are not independent training replicates. The frozen analysis must report seed-level paired deltas, confidence intervals, effect sizes, direction consistency, and multiplicity-controlled secondary tests for all preregistered contrasts. With very small seed counts, p-values are discrete and remain secondary evidence.

Development seed 1337 is excluded from confirmatory evidence after any tuning. Canonical confirmatory seeds are declared in `protocols/real_data_v21.json`.

## 12. Version identity

- Method: `OASIS-RC-v2.1`
- Implementation: `2.1.0-dev2`
- Checkpoint schema: `5`
- Experiment family: `oasis-rc-v2.1-gt-anchored-relational-energy-head`

`dev2` is a hardening revision: it keeps the OASIS-RC relation-energy hypothesis while strengthening N25 contracts, critic authorization, arbitrary-angle AOSK, evaluation separation, final-test control, gradient instrumentation, and confirmatory statistics.

A future confirmatory release must replace the development implementation version only after real-data N0/N25 execution, lambda calibration/freeze, all six arms, evaluation protocol, statistical plan, and the final-test bundle are frozen and verified.
