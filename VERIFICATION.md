# OASIS-RC-v2 repaired-branch verification status

Date: 2026-08-12

This document describes the current repaired implementation, not historical v2
source identity.

## Repository/CI checks

The GitHub Actions workflow performs:

```text
install from requirements-tested.txt
editable package install without dependency drift
python compileall: src + scripts + tests
pytest
bash -n: current smoke / lightweight smoke / three-seed / evaluation wrappers
```

A branch/commit is not considered code-ready unless its own workflow run passes;
a green run from an earlier commit is not transferable to a newer HEAD.

## Implemented scientific/reproducibility gates

```text
[implemented] explicit true-normal RGB identity and virtual zero masks
[implemented] crack rows cannot silently become normal because resized GT is empty
[implemented] raw/decoded RGB hashes
[implemented] raw/decoded binary-mask hashes
[implemented] decoded RGB-mask pair hashes
[implemented] split-independent lineage checks
[implemented] native-resolution mismatch requires alignment_verified=true
[implemented] crack-disappears-after-resize gate
[implemented] deterministic mixed crack/normal batches
[implemented] fixed optimizer-step budget across normal fractions
[implemented] isolated augmentation and RC-corruption RNG streams
[implemented] exact shared student-init checkpoint requirement
[implemented] shared frozen critic requirement for S1/S3
[implemented] true-normal critic valid/invalid semantics
[implemented] no-op corruption derives pair validity from actual invalid map
[implemented] AOSK independent of RC warmup/ramp
[implemented] epoch-mean auxiliary logging
[implemented] relation-energy diagnostics
[implemented] auxiliary gradient norm/alignment diagnostics
[implemented] normal_val FP diagnostics when lineage-safe normal_val exists
[implemented] student-only RGB inference contract
[implemented] evaluation resolution bound to checkpoint
[implemented] exact command/git/runtime/init/manifest/critic provenance
[implemented] strict deterministic mode for controlled runs
```

## Still requires local real-data/GPU validation

The repository does not contain the user's `/hdd1/...` datasets or A30 runtime,
so the following cannot truthfully be certified by GitHub CPU CI:

```text
[LOCAL REQUIRED] final real manifest Gate 0 at 256x256
[LOCAL REQUIRED] GT-only review for every alignment_verified=true mismatch
[LOCAL REQUIRED] parent/session lineage recovery for external normal patches if normal_val is desired
[LOCAL REQUIRED] A30 deterministic execution
[LOCAL REQUIRED] critic quality gate on real validation data
[LOCAL REQUIRED] four-arm 1-2 epoch GPU micro-smoke
[LOCAL REQUIRED] gradient diagnostics on trained critic
[LOCAL REQUIRED] validation-only N0/N25 controlled experiment
```

Canonical test must remain unopened until the protocol and hyperparameters are
frozen from validation.

## Provenance warning

The exact historical OASIS-RC-v2 source is missing. Current code implements
`OASIS-RC-v2-reconstructed`; it must not be described as bit-for-bit historical
reproduction. Historical reports/artifacts remain evidence of prior runs only.

## Method-level issue intentionally not changed by the repair

The reconstructed v2 definition uses:

```text
rank_gt = softplus(E_pred - E_gt + margin)
```

Changing its sign/margin semantics would change the scientific method rather
than repair an implementation bug. Any alternative formulation must be a named
v2.1/ablation and selected using validation only.
