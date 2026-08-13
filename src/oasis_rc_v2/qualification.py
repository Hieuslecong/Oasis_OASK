QUALIFICATION_THRESHOLDS = {
    "valid_crack_recall": 0.80,
    "invalid_recall": 0.90,
    "rgb_pair_drop": 0.05,
    "mask_pair_drop": 0.05,
}


def critic_gate_failures(metrics):
    failures = []
    for key, threshold in QUALIFICATION_THRESHOLDS.items():
        value = metrics.get(key)
        if value is None or float(value) < threshold:
            failures.append(f"{key}>={threshold:.2f}")
    if int(metrics.get("rgb_pair_samples", 0)) <= 0:
        failures.append("rgb_pair_samples>0")
    if int(metrics.get("mask_pair_samples", 0)) <= 0:
        failures.append("mask_pair_samples>0")
    if int(metrics.get("valid_crack_predictions", 0)) <= 0:
        failures.append("no_background_only_collapse")
    return failures


def critic_gate_passes(metrics):
    return not critic_gate_failures(metrics)
