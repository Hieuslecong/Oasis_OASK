# Phase 0 Repository Audit Report

**Project:** CRCV/OASIS/DP-GAN crack-segmentation stress framework  
**Audit date:** 2026-09-02  
**Audit scope:** actual repository state available in the current workspace  
**Audit status:** **BLOCKED — repository source is not available**

## Executive conclusion

The requested code audit cannot be completed from the current workspace because the target repository is not present. The working directory contains no source files, no valid Git repository, no dataset manifests, no model checkpoints, and no experiment configuration.

Therefore, the following claims are **not verified**:

- whether the current DP-GAN implementation matches the paper/source implementation;
- whether OASIS code exists in the target repository or is correctly integrated;
- whether the segmentation pipeline is executable and correct;
- whether dataset splits are valid and leakage-free;
- whether the existing baseline/smoke tests pass;
- whether any reported metric is reproducible from the current code.

No rewrite, refactor, synthetic data creation, checkpoint fabrication, or metric claim was made.

## Evidence collected

| Check | Result |
|---|---|
| Working directory | `/workspace/scratch/f63d05133465` |
| Repository files in working directory | None |
| Valid Git repository in working directory | No |
| Git branch/commit | Unavailable |
| Python project/config files | None in the working directory |
| Dataset directories/manifests | None found in the working directory |
| Checkpoints | None found in the working directory |
| Tests | None found in the working directory |
| Baseline/smoke execution | Not executable because source is absent |
| Repository retrieval | Public GitHub clone attempt was blocked by the available network/authentication path |

The repository names and branches known from earlier project context are not treated as the current source of truth. A valid audit must be performed against the actual checkout/commit supplied for this turn.

## A. Current architecture

**Status: BLOCKED.**

No implementation files were available, so the following architecture cannot be reconstructed or confirmed:

```text
dataset -> preprocessing -> segmentation / renderer -> stress generation
        -> geometry + realism validation -> hard search -> training -> evaluation
```

The intended target architecture from the specification is reasonable as a design constraint, but it is not evidence of the current repository architecture. It must not be reported as implemented.

## B. Modules reusable

**Status: UNVERIFIED.**

No modules can be classified as reusable without inspecting their source, imports, call sites, tests, and runtime behavior. In particular, the following remain unknown:

- DP-GAN generator/discriminator and checkpoint loader;
- OASIS implementation or adapter;
- dataset readers and transforms;
- segmentation models and loss functions;
- training/evaluation/checkpoint code;
- logging and experiment tracking;
- existing leakage or reproducibility utilities.

## C. Modules incorrect

**Status: UNVERIFIED.**

No code was available for correctness review. It would be scientifically invalid to label a module incorrect or correct based only on the requested target design or historical reports.

## D. Dead code

**Status: UNVERIFIED.**

Dead-code detection requires at least the source tree, package entry points, import graph, scripts, and configuration references. No deletion is authorized before that analysis because the prompt requires preserving reproducibility until obsolete code is demonstrated.

## E. Duplicate code

**Status: UNVERIFIED.**

No source tree was available for duplicate helper, model, loss, dataset, or preprocessing comparison.

## F. Data leakage risks

**Status: BLOCKED — no data or manifests available.**

The required checks could not be run:

- duplicate filenames;
- exact image hashes;
- perceptual duplicates;
- crop descendants;
- same source image across splits;
- synthetic source-mask relationships;
- train/test overlap;
- target-domain participation in calibration or tuning.

The mandatory leakage audit script was not created because the dataset interface and repository conventions are unknown. Creating a generic script now would risk silently using the wrong image-mask pairing or split semantics.

**Experiment stop rule:** all experiments that depend on the missing repository/data must remain stopped. No metric should be treated as valid until the leakage audit produces an explicit pass.

## G. Reproducibility risks

The current workspace cannot provide evidence for:

- pinned dependencies;
- Python/PyTorch/CUDA versions;
- seed handling;
- deterministic settings;
- dataset split hashes;
- Git commit and dirty-state logging;
- checkpoint/resume completeness;
- AMP behavior;
- hardware and device selection.

These are **open risks**, not confirmed defects in the absent repository.

## H. Missing experiments

Because the implementation and data are absent, every experiment below is pending rather than failed:

1. baseline segmentation integrity;
2. real nuisance extraction and calibration;
3. DP-GAN renderer qualification;
4. factorized nuisance-control validation;
5. structural validity/envelope validation;
6. real-calibrated random-stress baseline;
7. unconstrained hard search;
8. constrained hard search;
9. multi-backbone evaluation;
10. cross-domain and leave-one-dataset-out evaluation;
11. single-factor robustness profiles and AUC statistics;
12. multi-nuisance stress analysis;
13. at least three independent seeds for final experiments;
14. computational-cost profiling;
15. failure analysis and visual QA.

## I. Proposed minimal implementation plan

The following plan is intentionally conditional. It must begin only after a real checkout is available.

### Phase 0 — Repository and data intake

1. Provide or mount the exact repository.
2. Record remote URL, branch, commit, dirty status, and submodules.
3. Inventory source files, configs, scripts, tests, datasets, and checkpoints.
4. Run the audit against the actual HEAD.
5. Stop if dataset leakage is detected.

### Phase 1 — Baseline integrity

Reuse existing abstractions where correct. Add only the minimum shared interfaces required for:

- dataset registry and canonical sample schema;
- model-agnostic segmenter;
- metrics and numerical-stability tests;
- checkpoint/resume;
- environment, seed, config, and split-hash logging.

Do not add stress search until a real-data baseline passes smoke and overfit tests.

### Phase 2 — Real nuisance calibration

Implement background-first feature extraction, then estimate empirical distributions, quantiles, robust statistics, and covariance using training domains only. Save a versioned calibration artifact and calibration-data hash.

### Phase 3 — Renderer adapter and qualification

Inspect the existing DP-GAN implementation before changing it. If it is correct and usable, wrap it behind a renderer interface. If it only exposes latent noise, label it **non-factorized** and retain it as G0. Do not claim explicit nuisance control until it is implemented and validated.

Qualification must test semantic preservation, realism, nuisance response, independent factor variation, and avoidance of label corruption.

### Phase 4 — Factorized nuisance control

Add the smallest compatible conditioning mechanism, preferably conditional affine/FiLM-style modulation where appropriate. Compare:

- G0: original latent DP-GAN;
- G1: explicit nuisance conditioning;
- G2: conditioning plus justified regularization.

Retain G2 only if independent-control evidence supports it.

### Phase 5 — Structural validity

Implement geometry checks for skeleton, width, components, junctions, connectivity, and optional topology. Derive or validate thresholds on a validation set; never tune them on unseen test domains.

### Phase 6 — Random calibrated stress

Train the required real-plus-random-real-calibrated-stress baseline. Record candidate validity, realism, and geometry diagnostics separately.

### Phase 7 — Unconstrained hard search

Implement random candidate search first. Freeze the segmenter during the inner maximization and update the segmenter only after selecting a candidate. Keep hardness, validity, and realism as separate quantities.

### Phase 8 — Constrained hard search

Compare constrained and unconstrained search under identical data, seeds, budgets, and backbone policies. Reject the method if hardest samples are mostly generator artifacts or invalid geometry.

### Phase 9 — Multi-model evaluation

Run U-Net, SegFormer, and one feasible strong crack-specific model with fair training and evaluation policies.

### Phase 10 — Cross-domain evaluation

Use source-only calibration and tuning for unseen targets. Run both source-to-unseen and LODO protocols when the available datasets support them.

### Phase 11 — Robustness profiling

Profile each active nuisance factor independently, then selected correlated combinations. Save curves, normalized AUC/statistics, and failure examples.

### Phase 12 — Final freeze and validation

Freeze code and configs before final test evaluation. Run tests, leakage audit, smoke test, overfit test, resume test, renderer qualification, ablations, multi-seed runs, cost profiling, and final validation reporting.

## Required repository intake before continuing

At least one of the following is required:

- a local repository directory mounted in the workspace;
- a valid archive containing the repository;
- an accessible Git remote/connector with the exact repository and target branch.

The intake must also identify actual dataset roots and available checkpoints. Missing datasets/checkpoints may remain declared dependencies, but the source code must be available before code-level audit conclusions are made.

## Current scientific decision

**INCONCLUSIVE, not PASS.**

The research direction cannot be accepted or rejected from this workspace. The correct next action is to supply the actual repository checkout, then rerun Phase 0 against its real HEAD. No hard-search implementation should begin before that audit is complete.
