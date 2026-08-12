# Historical source gap

## Finding

The current workspace contains the canonical OASIS-RC source, multiple v2
checkpoints/result directories, and reports describing OASIS-RC-v2. It does not
contain a separately versioned historical source file implementing the exact
v2 corrupted-mask ranking experiment.

The file `oasis_cycle_aosk/train_v2.py` in the parent legacy project is not
OASIS-RC-v2. It belongs to an older generator/discriminator cycle experiment
and must not be used as evidence for OASIS-RC-v2.

## Consequence

Historical v2 metrics can be audited as recorded evidence, but cannot be
reproduced bit-for-bit from a verified source hash. The implementation included
here is reconstructed from the audited method definition:

- v1 GT-vs-prediction relational ranking;
- explicit prediction-vs-corrupted-mask ranking;
- GT-background-only false-positive penalty;
- optional critic pair-consistency weight.

## Required wording

Use:

> OASIS-RC-v2 was reconstructed from the surviving audited specification and
> result artifacts; the exact historical source snapshot was unavailable.

Do not use:

> The package exactly reproduces the historical OASIS-RC-v2 implementation.

## Closure condition

The gap is closed only if the historical v2 source is recovered with a
verifiable source-tree hash and matches the saved checkpoint architecture and
training metadata. Until then, new runs belong to the reconstructed v2 lineage.
