# Package verification

Date: 2026-08-12

```text
Python compileall: PASS
Unit tests: 7 passed
Reconstructed connected forward/backward smoke: PASS
Critic gate during verification: PASS
Student-only checkpoint isolation: PASS
Exact historical v2 source identity: MISSING
Raw image data: not included
Strict source-disjoint certification of bundled debug manifest: FAIL
```

The reconstructed smoke produced finite `rank_gt`, `rank_corrupted` and
background FP terms. The saved checkpoint keys were:

```text
config, inference_contract, lambda_oasis, method_version, mode, student,
student_kind, student_width, threshold_validation
```

No critic or AOSK state was present. This verifies execution of the
reconstructed implementation, not exact reproduction of historical v2 metrics.
