#!/usr/bin/env python3
"""Confirmatory paired statistics for frozen OASIS-RC-v2.1 results.

The sampling unit for method uncertainty is the training seed, not an image
inside one trained checkpoint. Input is an immutable result index mapping each
arm and seed to a crack-evaluation JSON and a normal-evaluation JSON.
Image-level rows remain available in those files for descriptive/error analysis
but are not treated as independent training replicates here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONTRASTS = {
    "B1-B0": ("B0", "B1"),
    "B2-B0": ("B0", "B2"),
    "S1-B0": ("B0", "S1"),
    "S2-B0": ("B0", "S2"),
    "S3-S2": ("S2", "S3"),
}
METRICS = {
    "accuracy": ("dice", "iou"),
    "topology": ("cldice", "mean_component_excess"),
    "robustness": (
        "normal_any_fp_rate",
        "normal_fp_pixels_mean",
        "normal_fp_components_mean",
    ),
}
LOWER_IS_BETTER = {
    "mean_component_excess",
    "normal_any_fp_rate",
    "normal_fp_pixels_mean",
    "normal_fp_components_mean",
}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _validate_eval(path, population):
    data = _read_json(path)
    if population == "crack":
        if int(data.get("crack_image_count", 0)) <= 0:
            raise ValueError(f"{path}: crack evaluation contains no crack-positive rows")
    elif population == "normal":
        if int(data.get("normal_image_count", 0)) <= 0:
            raise ValueError(f"{path}: normal evaluation contains no true-negative rows")
    else:
        raise ValueError(f"unknown population {population!r}")
    return data


def _metric_from_record(record, metric):
    population = "normal" if metric.startswith("normal_") else "crack"
    path = record.get(population)
    if not path:
        raise ValueError(
            f"confirmatory record missing required {population} evaluation for metric {metric}"
        )
    data = _validate_eval(path, population)
    if metric not in data or data.get(metric) is None:
        raise ValueError(f"{path}: missing required confirmatory metric {metric!r}")
    return float(data[metric])


def _bootstrap_seed_mean(delta, seed=20260821, reps=20000):
    delta = np.asarray(delta, dtype=float)
    if delta.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for start in range(0, reps, 1000):
        n = min(1000, reps - start)
        idx = rng.integers(0, delta.size, size=(n, delta.size))
        means[start : start + n] = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def _exact_sign_flip_p(delta):
    """Two-sided exact paired randomisation p-value for small seed counts."""
    delta = np.asarray(delta, dtype=float)
    nonzero = delta[np.abs(delta) > 0]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    if n > 20:
        return None
    observed = abs(float(nonzero.mean()))
    exceed = 0
    total = 1 << n
    for bits in range(total):
        signs = np.fromiter(
            (1.0 if (bits >> i) & 1 else -1.0 for i in range(n)),
            dtype=float,
            count=n,
        )
        if abs(float((nonzero * signs).mean())) >= observed - 1e-15:
            exceed += 1
    return float(exceed / total)


def _summarize(delta, seeds, bootstrap_reps):
    d = np.asarray(delta, dtype=float)
    if d.size == 0:
        return None
    std = float(d.std(ddof=1)) if d.size > 1 else 0.0
    dz = None if d.size < 2 or std == 0.0 else float(d.mean() / std)
    return {
        "n_seeds": int(d.size),
        "seeds": [int(s) for s in seeds],
        "seed_deltas": [float(x) for x in d],
        "mean_delta": float(d.mean()),
        "std_delta": std,
        "median_delta": float(np.median(d)),
        "bootstrap_95ci_seed_mean": _bootstrap_seed_mean(d, reps=bootstrap_reps),
        "positive_seed_fraction": float((d > 0).mean()),
        "negative_seed_fraction": float((d < 0).mean()),
        "cohen_dz": dz,
        "exact_sign_flip_p_two_sided": _exact_sign_flip_p(d),
    }


def _holm_adjust(items):
    """Holm-adjust p-values in-place for one preregistered metric family."""
    usable = [(key, value) for key, value in items if value is not None]
    usable.sort(key=lambda kv: kv[1])
    m = len(usable)
    adjusted = {}
    running = 0.0
    for rank, (key, p) in enumerate(usable):
        candidate = min(1.0, (m - rank) * float(p))
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def analyze(index, bootstrap_reps):
    arms = index.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("result index requires an arms mapping")
    required_arms = {arm for pair in CONTRASTS.values() for arm in pair}
    missing = sorted(required_arms - set(arms))
    if missing:
        raise ValueError("result index missing arms: " + ", ".join(missing))

    declared_seeds = [int(s) for s in index.get("seeds", [])]
    if not declared_seeds:
        common = None
        for arm in required_arms:
            keys = {int(s) for s in arms[arm]}
            common = keys if common is None else common & keys
        declared_seeds = sorted(common or [])
    if len(declared_seeds) < 2:
        raise ValueError("paired seed analysis requires at least two complete seeds")
    if len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError("declared seeds contain duplicates")

    results = {}
    p_by_family = {family: [] for family in METRICS}
    for contrast, (base_arm, treat_arm) in CONTRASTS.items():
        c = {"base": base_arm, "treatment": treat_arm, "metrics": {}}
        for family, metrics in METRICS.items():
            for metric in metrics:
                deltas = []
                used_seeds = []
                for seed in declared_seeds:
                    base = arms[base_arm].get(str(seed), arms[base_arm].get(seed))
                    treat = arms[treat_arm].get(str(seed), arms[treat_arm].get(seed))
                    if not base or not treat:
                        raise ValueError(
                            f"missing paired result for {contrast}, seed={seed}"
                        )
                    b = _metric_from_record(base, metric)
                    t = _metric_from_record(treat, metric)
                    delta = b - t if metric in LOWER_IS_BETTER else t - b
                    if not np.isfinite(delta):
                        raise ValueError(
                            f"non-finite paired delta for {contrast}, metric={metric}, seed={seed}"
                        )
                    deltas.append(delta)
                    used_seeds.append(seed)
                if used_seeds != declared_seeds:
                    raise RuntimeError(
                        f"internal pairing error for {contrast}, metric={metric}: "
                        f"used={used_seeds}, declared={declared_seeds}"
                    )
                summary = _summarize(deltas, used_seeds, bootstrap_reps)
                if summary is None or summary["n_seeds"] != len(declared_seeds):
                    raise RuntimeError(
                        f"confirmatory metric lost seed pairs: {contrast}/{metric}"
                    )
                summary["family"] = family
                summary["higher_delta_means"] = "treatment_better"
                c["metrics"][metric] = summary
                p_by_family[family].append(
                    ((contrast, metric), summary["exact_sign_flip_p_two_sided"])
                )
        results[contrast] = c

    for family, pairs in p_by_family.items():
        adjusted = _holm_adjust(pairs)
        for (contrast, metric), p_adj in adjusted.items():
            results[contrast]["metrics"][metric]["holm_p_within_family"] = p_adj

    return {
        "schema": "oasis-rc-v2.1-paired-seed-stats-v2",
        "sampling_unit": "training_seed",
        "declared_seeds": declared_seeds,
        "n_declared_seeds": len(declared_seeds),
        "bootstrap_reps": int(bootstrap_reps),
        "contrasts": results,
        "primary_interpretation": (
            "Use paired seed deltas, confidence intervals, effect sizes and direction "
            "consistency. With very small seed counts, exact p-values are discrete and "
            "must remain secondary evidence."
        ),
        "multiplicity": "Holm correction within each preregistered metric family",
        "population_rule": (
            "accuracy/topology from crack evaluation only; robustness from normal "
            "evaluation only; never average crack overlap metrics over true-negative rows"
        ),
        "complete_case_rule": (
            "Every preregistered metric must exist and be finite for every declared paired seed; "
            "missing metrics fail closed rather than reducing n post hoc"
        ),
        "warning": (
            None
            if len(declared_seeds) >= 5
            else "Fewer than five complete seeds: development/descriptive evidence only"
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index",
        required=True,
        help=(
            "JSON: {seeds:[...], arms:{B0:{seed:{crack:path,normal:path}}, ...}}"
        ),
    )
    p.add_argument("--bootstrap-reps", type=int, default=20000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.bootstrap_reps < 1000:
        raise ValueError("--bootstrap-reps must be >=1000")
    index = _read_json(a.index)
    result = analyze(index, a.bootstrap_reps)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
