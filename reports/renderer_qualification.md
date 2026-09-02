# Renderer qualification

Status: **BLOCKED for DP-GAN; PASS for toy integration only**.

No compatible DP-GAN checkpoint or verified backend is present. The adapter
therefore refuses to claim factorized control. `ToyStressRenderer` passes the
geometry-preservation integration test and single-factor toy sweep, but its
outputs are not evidence of DP-GAN realism or scientific success.
