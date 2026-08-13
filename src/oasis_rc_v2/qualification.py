QUALIFICATION_MINIMUMS = {
    "valid_crack_recall": 0.80,
    "invalid_recall": 0.90,
    "rgb_pair_drop": 0.05,
    "mask_pair_drop": 0.05,
    "valid_normal_bg_recall": 0.80,
    "normal_pair_valid_mean": 0.50,
    "min_corruption_invalid_recall": 0.70,
}

QUALIFICATION_MAXIMUMS = {
    "normal_invalid_rate": 0.20,
}


def critic_gate_failures(metrics):
    failures = []
    for key, threshold in QUALIFICATION_MINIMUMS.items():
        value = metrics.get(key)
        if value is None or float(value) < threshold:
            failures.append(f"{key}>={threshold:.2f}")
    for key, threshold in QUALIFICATION_MAXIMUMS.items():
        value = metrics.get(key)
        if value is None or float(value) > threshold:
            failures.append(f"{key}<={threshold:.2f}")
    if int(metrics.get("rgb_pair_samples", 0)) <= 0:
        failures.append("rgb_pair_samples>0")
    if int(metrics.get("mask_pair_samples", 0)) <= 0:
        failures.append("mask_pair_samples>0")
    if int(metrics.get("normal_samples", 0)) <= 0:
        failures.append("normal_samples>0")
    if int(metrics.get("valid_crack_predictions", 0)) <= 0:
        failures.append("no_background_only_collapse")
    per_kind = metrics.get("corruption_invalid_recall", {})
    for name in (
        "C1_translation",
        "C2_erosion",
        "C3_dilation",
        "C4_local_break",
        "C5_wrong_width",
        "C6_wrong_connection",
        "C7_donor_mask",
        "C8_crack_on_normal",
        "C9_texture_fp_blob",
    ):
        value = per_kind.get(name)
        if value is None:
            failures.append(f"{name}:samples>0")
    return failures


def critic_gate_passes(metrics):
    return not critic_gate_failures(metrics)
