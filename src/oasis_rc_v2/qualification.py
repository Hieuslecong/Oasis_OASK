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


def critic_gate_failures(metrics):
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
    required_corruptions = list(BASE_CORRUPTIONS)
    if normal_expected:
        required_corruptions.append("C8_crack_on_normal")
    for name in required_corruptions:
        if per_kind.get(name) is None:
            failures.append(f"{name}:samples>0")
    return failures


def critic_gate_passes(metrics):
    return not critic_gate_failures(metrics)
