BASE_MINIMUMS = {
    "valid_crack_recall": 0.80,
    "invalid_recall": 0.90,
    "rgb_pair_drop": 0.05,
    "mask_pair_drop": 0.05,
    "min_corruption_invalid_recall": 0.70,
}

NORMAL_MINIMUMS = {
    "valid_normal_bg_recall": 0.80,
    "normal_pair_valid_mean": 0.50,
}

NORMAL_MAXIMUMS = {
    "normal_invalid_rate": 0.20,
}

# Development-only defaults. They are intentionally modest and MUST be frozen
# from development evidence before any confirmatory run. They test whether an
# energy landscape exists at all; they are not paper significance thresholds.
ENERGY_DEV_MINIMUMS = {
    "positive_energy_gap_fraction": 0.70,
    "continuous_path_order_fraction": 0.65,
}
ENERGY_DEV_POSITIVE = (
    "median_energy_gap",
    "mean_energy_gap",
)
ENERGY_MIN_SAMPLES = 16

BASE_CORRUPTIONS = (
    "C1_translation",
    "C2_erosion",
    "C3_dilation",
    "C4_local_break",
    "C5_wrong_width",
    "C6_wrong_connection",
    "C7_donor_mask",
    "C9_texture_fp_blob",
)

MIN_SAMPLES_PER_CORRUPTION = 16


def critic_gate_failures(metrics):
    """Representation/classification gate retained from v2.0.4."""
    failures = []
    normal_expected = bool(
        metrics.get(
            "normal_supervision_expected",
            int(metrics.get("normal_samples", 0)) > 0,
        )
    )

    for key, threshold in BASE_MINIMUMS.items():
        value = metrics.get(key)
        if value is None or float(value) < threshold:
            failures.append(f"{key}>={threshold:.2f}")

    if normal_expected:
        for key, threshold in NORMAL_MINIMUMS.items():
            value = metrics.get(key)
            if value is None or float(value) < threshold:
                failures.append(f"{key}>={threshold:.2f}")
        for key, threshold in NORMAL_MAXIMUMS.items():
            value = metrics.get(key)
            if value is None or float(value) > threshold:
                failures.append(f"{key}<={threshold:.2f}")
        if int(metrics.get("normal_samples", 0)) <= 0:
            failures.append("normal_samples>0")

    if int(metrics.get("rgb_pair_samples", 0)) <= 0:
        failures.append("rgb_pair_samples>0")
    if int(metrics.get("mask_pair_samples", 0)) <= 0:
        failures.append("mask_pair_samples>0")
    if int(metrics.get("valid_crack_predictions", 0)) <= 0:
        failures.append("no_background_only_collapse")

    per_kind = metrics.get("corruption_invalid_recall", {})
    per_kind_samples = metrics.get("corruption_samples", {})
    required_corruptions = list(BASE_CORRUPTIONS)
    if normal_expected:
        required_corruptions.append("C8_crack_on_normal")
    for name in required_corruptions:
        if per_kind.get(name) is None:
            failures.append(f"{name}:samples>0")
        if int(per_kind_samples.get(name, 0)) < MIN_SAMPLES_PER_CORRUPTION:
            failures.append(f"{name}:samples>={MIN_SAMPLES_PER_CORRUPTION}")
    return failures


def critic_gate_passes(metrics):
    return not critic_gate_failures(metrics)


def relation_energy_gate_failures(metrics):
    """v2.1 usability gate for the relation-energy landscape.

    This gate deliberately does not substitute for student-gradient diagnostics.
    It establishes that the critic orders GT, continuous soft corruption paths,
    and structured corruptions in a direction compatible with the v2.1 loss.
    Gradient norm/alignment remains a required pre-confirmatory diagnostic.
    """
    failures = []
    if int(metrics.get("energy_samples", 0)) < ENERGY_MIN_SAMPLES:
        failures.append(f"energy_samples>={ENERGY_MIN_SAMPLES}")

    for key, threshold in ENERGY_DEV_MINIMUMS.items():
        value = metrics.get(key)
        if value is None or float(value) < threshold:
            failures.append(f"{key}>={threshold:.2f}")

    for key in ENERGY_DEV_POSITIVE:
        value = metrics.get(key)
        if value is None or float(value) <= 0.0:
            failures.append(f"{key}>0")

    if metrics.get("energy_finite") is not True:
        failures.append("energy_finite=true")
    return failures


def relation_energy_gate_passes(metrics):
    return not relation_energy_gate_failures(metrics)


def connected_gate_failures(representation_metrics, energy_metrics):
    return critic_gate_failures(representation_metrics) + relation_energy_gate_failures(
        energy_metrics
    )


def connected_gate_passes(representation_metrics, energy_metrics):
    return not connected_gate_failures(representation_metrics, energy_metrics)
