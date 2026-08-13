# OmniCrack30k-CleanEval-v1 — protocol status

**Status:** previous frozen snapshot is **superseded** and must not be used for
paper metrics or final test.

The 2026-08-13 snapshot mixed source×split sampling decisions with row-level
claims, removed evaluation rows for some lineage conflicts, and contained
inconsistent frozen SHA256 values. Those artifacts remain in `reports/` only as
historical audit evidence.

The authoritative construction path is now:

```text
canonical manifest
  -> scripts/clean_manifest.py
       preserve test > val > train for cross-split leakage
  -> scripts/build_cleaneval_v1.py
       explicit row-level N0 only
       N1/N2/N3/unreviewed native-empty targets excluded
       one-run Gate 0 + SHA freeze
  -> scripts/add_normal_rgb_to_manifest.py
  -> Gate 0 with --normal-policy train
  -> critic qualification
  -> validation-only S0/S1/S2/S3
```

`run_validation_eval.sh` is validation-only. Canonical test can only be opened
through `run_final_test.sh` with an explicit `PROTOCOL_LOCK.json`.

For the local A30 workflow use `scripts/run_training_ready.sh`. It rebuilds the
derived benchmark and critic from the exact manifest before launching the four
validation arms. No historical `benchmark_freeze.json` under
`reports/cleaneval_v1/` should be treated as authoritative after this change;
the new build writes its own freeze file into the experiment output directory.
