# Phase 0 â Repository Audit

**Repository:** `Hieuslecong/Oasis_OASK`  
**Audited ref:** `main`  
**Audited commit:** `1b14213de4121c99cd87f184c453fcbe3c1bb1e8`  
**Date:** 2026-09-02  
**Status:** **PASS for static/unit audit; BLOCKED for real-data and DP-GAN qualification**

## Executive conclusion

The repository is an OASIS-RC-v2/AOSK crack-segmentation research package. It is not a DP-GAN repository and it does not contain a DP-GAN generator/discriminator, a nuisance calibration module, a stress renderer adapter, a structural validity envelope, or hard nuisance search.

The existing implementation is useful as a reference segmentation/training package, but the requested DP-GAN stress-learning framework must be added as an independent, minimal package. Existing OASIS-RC/AOSK code must not be silently re-labelled as DP-GAN or overwritten.

## Evidence and checks

| Check | Result |
|---|---|
| Main source tree | 50 text files materialized for audit; binary artifacts excluded from local execution |
| Main commit | `1b14213de4121c99cd87f184c453fcbe3c1bb1e8` |
| Static compilation | PASS |
| Existing unit tests | PASS: 26/26 on CPU, PyTorch 2.5.1+cpu |
| Shell syntax checks | PASS |
| Real manifest Gate 0 | FAIL: 90 rows reference missing external image paths |
| Real images/masks | Not available in workspace |
| DP-GAN checkpoint | Not available in workspace/repository |
| GPU | Not available in ChatGPT runtime; CPU only |
| Canonical test evaluation | Not opened |

## A. Current architecture

```text
manifest JSONL
    -> ManifestDataset
    -> lightweight RGB student
    -> BCE/Dice segmentation loss
    -> optional OASIS-RC-v2 relational critic
    -> optional AOSK loss
    -> validation threshold and student-only checkpoint
```

Current source areas:

- `src/oasis_cycle_aosk/data.py`: manifest-backed RGB/mask loader;
- `src/oasis_cycle_aosk/audit.py`: metadata, hash, lineage and resize checks;
- `src/oasis_cycle_aosk/models.py`: lightweight segmenters and relational critic;
- `src/oasis_cycle_aosk/losses*.py`: segmentation, RC and AOSK losses;
- `src/oasis_cycle_aosk/train_oasis_rc_v2.py`: controlled four-arm training;
- `src/oasis_cycle_aosk/evaluate_rc.py`: RGB-only student evaluation;
- `scripts/`: smoke, three-seed, normal-RGB and provenance utilities;
- `tests/`: architecture, audit, normal-RGB, training-contract and loss tests.

The requested target architecture will be added separately:

```text
crack_stress/
    datasets, models, calibration, renderer adapter,
    geometry/realism envelope, random sampler, hard search,
    metrics, training and evaluation utilities
```

## B. Modules reusable

Reusable with review and adapters:

1. `ManifestDataset` sample loading and explicit true-normal semantics.
2. Existing lineage/hash audit ideas.
3. Lightweight segmentation models as optional baselines.
4. Existing segmentation loss implementation as a reference, not as a new contribution.
5. Existing student-only inference contract.
6. Existing seed/provenance conventions.
7. Existing tests for frozen critic, gradients, and deployment separation.

The OASIS-RC critic is not reused as the DP-GAN renderer. It remains a separate historical/reference method.

## C. Modules incorrect or insufficient for the requested method

These are insufficiencies relative to the DP-GAN stress prompt, not claims that all existing code is generally defective:

- no DP-GAN adapter or verified DP-GAN checkpoint loader;
- no explicit nuisance vector or real calibration model;
- no factorized conditioning or G0/G1/G2 qualification protocol;
- no geometry validator comparing generated semantic geometry with the input mask;
- no realism validator with a separately reported score;
- no random real-calibrated sampler or hard nuisance search;
- no freeze-segmenter inner maximization implementation;
- no cross-domain/LODO runner for the requested framework;
- no required precision/recall/mIoU/clDice/thin-crack metric package;
- no complete resume checkpoint contract for the requested training loop;
- no required final reports and experiment registry.

## D. Dead code and repository hygiene

No deletion was performed. The upstream tree contains committed Python cache files and many historical artifacts/checkpoints. They should be reviewed in a later cleanup commit, but not deleted before provenance is mapped.

The legacy OASIS-RC files remain valuable as reference evidence and must not be treated as DP-GAN implementation.

## E. Duplicate code

There are parallel legacy and v2 training/loss entry points (`train_oasis_rc.py`, `train_oasis_rc_v2.py`, `losses.py`, `losses_v2.py`). They may be intentional historical variants, so automatic deletion would be unsafe. The new framework will not duplicate their internals; it will expose a narrow segmenter/loss interface.

## F. Data leakage risks

The current manifest uses paths under `/workspace/scratch/4ab02dda3b35/pilot_real_debug/real_debug_data`, which are not present in the current workspace. Consequently, real leakage status is **unverified**, not clean.

Existing audit coverage is useful but does not fully implement the requested checks for:

- perceptual duplicates;
- crop-descendant detection;
- synthetic source-mask relationships;
- complete train/test overlap evidence across all source forms.

No real experiment may be accepted until a source-disjoint manifest and leakage report are produced from actual data.

## G. Reproducibility risks

- The historical OASIS-RC-v2 source is explicitly documented as reconstructed.
- Existing manifests are environment-specific and not usable in this workspace.
- DP-GAN source/checkpoint provenance is absent from this repository.
- The current package records useful provenance, but the new framework needs config hash, split hash, calibration hash, renderer checkpoint hash, RNG state, and complete optimizer/scheduler/scaler state.
- CPU-only execution prevents GPU performance claims.

## H. Missing experiments

Pending for the requested DP-GAN method:

1. real nuisance extraction/calibration;
2. G0 original DP-GAN latent baseline;
3. G1 explicit nuisance conditioning;
4. G2 conditioning plus regularization;
5. semantic-preservation/geometry qualification;
6. independent factor-control validation;
7. realism qualification;
8. random calibrated stress baseline;
9. unconstrained hard search;
10. constrained hard search;
11. U-Net, SegFormer and crack-specific baseline comparison;
12. cross-domain/LODO evaluation;
13. single-factor robustness curves and AUC statistic;
14. multi-seed final experiments;
15. compute-cost and failure-analysis reports.

## I. Minimal implementation plan

1. Preserve current OASIS-RC/AOSK package unchanged as reference.
2. Add a small `crack_stress` package with dynamic nuisance vectors, manifest registry, calibration, metrics, geometry/realism envelope, renderer protocol, DP-GAN adapter, random sampler and random-candidate hard search.
3. Add only the required CLI scripts and YAML configs.
4. Add correctness tests for empty/all-crack/one-pixel/disconnected masks, determinism, invalid geometry rejection, candidate selection, checkpoint/resume and inference separation.
5. Run CPU unit tests and toy integration only for implementation verification.
6. Run real-data Gate 0 and calibration only after actual datasets are supplied.
7. Run DP-GAN qualification only after a verified DP-GAN checkpoint and compatible preprocessing are supplied.
8. Run full training only after leakage, renderer qualification, smoke, overfit and resume gates pass.
9. Push only reviewed source/config/tests/docs; never push datasets, checkpoints, caches or generated temporary files.

## Scientific decision

The existing OASIS-RC package passes static/unit verification but does not yet implement the requested DP-GAN stress-learning method. The DP-GAN hypotheses are **INCONCLUSIVE**. The correct implementation strategy is an isolated renderer-agnostic extension, with DP-GAN treated as an unverified backend until qualification succeeds.

## Implementation checkpoint after audit

The minimal isolated `src/crack_stress/` extension has now been implemented
without replacing the OASIS-RC/AOSK package. It includes the dataset registry,
dynamic nuisance vector, real-background calibration, metrics, renderer
protocol and G0/G1-compatible DP-GAN adapter, geometry/realism gates, random
sampling, frozen-segmenter hard candidate search, checkpointing, CLI scripts,
configs and tests. CPU validation is 35/35 tests, plus toy smoke, overfit and
resume checks. This does not clear the real-data or DP-GAN qualification
blockers described above.
