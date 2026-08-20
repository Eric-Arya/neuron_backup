#!/usr/bin/env python3
"""Paired statistical analysis of the fixed-neuron safety/capability sweep."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata
from scipy.stats import chi2


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    base = root / "results/tradeoff/fixed_top25_strength_sweep"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--safety-dir", type=Path, default=base / "harmbehavior_first100"
    )
    parser.add_argument(
        "--capability-dir", type=Path, default=base / "gsm8k_first100"
    )
    parser.add_argument("--output-dir", type=Path, default=base / "analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutation-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=112)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return 100 * (center - half), 100 * (center + half)


def paired_effect(
    baseline: np.ndarray,
    condition: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, float | int]:
    if len(baseline) != len(condition):
        raise ValueError("Paired arrays have different lengths")
    delta = condition.astype(float) - baseline.astype(float)
    gained = int(np.sum((~baseline) & condition))
    lost = int(np.sum(baseline & (~condition)))
    discordant = gained + lost
    p_value = float(binomtest(gained, discordant, 0.5).pvalue) if discordant else 1.0
    indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
    boot = 100 * delta[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "delta_pp": float(100 * delta.mean()),
        "bootstrap_ci_low_pp": float(low),
        "bootstrap_ci_high_pp": float(high),
        "negative_to_positive": gained,
        "positive_to_negative": lost,
        "discordant": discordant,
        "mcnemar_exact_p": p_value,
    }


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def exact_spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Exact two-sided permutation p-value; intended for the small strength grid."""
    xr = rankdata(x)
    yr = rankdata(y)
    observed = pearson(xr, yr)
    if math.isnan(observed):
        return observed, math.nan
    extreme = 0
    total = 0
    for permutation in itertools.permutations(yr.tolist()):
        correlation = pearson(xr, np.asarray(permutation))
        extreme += abs(correlation) >= abs(observed) - 1e-12
        total += 1
    return observed, extreme / total


def cochran_q(matrix: np.ndarray) -> tuple[float, float]:
    """Omnibus test for matched binary outcomes across all strengths."""
    values = matrix.astype(float)
    k = values.shape[1]
    column_totals = values.sum(axis=0)
    row_totals = values.sum(axis=1)
    total = column_totals.sum()
    denominator = k * total - np.square(row_totals).sum()
    if denominator == 0:
        return math.nan, math.nan
    statistic = (k - 1) * (k * np.square(column_totals).sum() - total * total) / denominator
    return float(statistic), float(chi2.sf(statistic, k - 1))


def paired_permutation_trend(
    strengths: np.ndarray,
    matrix: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    """Two-sided within-example permutation test for a linear dose trend."""
    centered = strengths - strengths.mean()
    denominator = np.square(centered).sum()
    observed_rates = matrix.mean(axis=0)
    observed_slope = float(np.dot(centered, observed_rates) / denominator * 100)
    extreme = 0
    for _ in range(samples):
        orders = np.argsort(rng.random(matrix.shape), axis=1)
        permuted = np.take_along_axis(matrix, orders, axis=1).mean(axis=0)
        slope = float(np.dot(centered, permuted) / denominator * 100)
        extreme += abs(slope) >= abs(observed_slope) - 1e-12
    # Add one to numerator and denominator for a valid Monte Carlo p-value.
    return observed_slope, (extreme + 1) / (samples + 1)


def safety_data(path: Path) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    rows = read_jsonl(path / "generations.jsonl")
    frame = pd.DataFrame(rows).sort_values(["epsilon", "id"])
    conditions: dict[float, np.ndarray] = {}
    expected_ids = None
    for strength, group in frame.groupby("epsilon", sort=True):
        ids = group["id"].astype(int).tolist()
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"Safety IDs do not match at strength {strength}")
        conditions[float(strength)] = group["jailbroken"].astype(bool).to_numpy()
    baseline = (
        frame[frame["epsilon"] == min(conditions)]["baseline_jailbroken"]
        .astype(bool)
        .to_numpy()
    )
    if not np.array_equal(conditions[min(conditions)], baseline):
        raise ValueError("The no-op safety condition does not reproduce baseline labels")
    return frame, conditions


def capability_data(path: Path) -> tuple[dict[float, pd.DataFrame], dict[float, np.ndarray]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    frames: dict[float, pd.DataFrame] = {}
    conditions: dict[float, np.ndarray] = {}
    names = [(0.0, "baseline")]
    names.extend(
        (float(strength), f"strength_{strength:g}".replace(".", "p"))
        for strength in summary["controller"]["strengths"]
    )
    expected_ids = None
    for strength, name in names:
        frame = pd.DataFrame(read_jsonl(path / f"{name}_responses.jsonl")).sort_values("id")
        ids = frame["id"].astype(int).tolist()
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"GSM8K IDs do not match at strength {strength}")
        frames[strength] = frame
        conditions[strength] = frame["correct"].astype(bool).to_numpy()
    return frames, conditions


def write_plot(summary: pd.DataFrame, path: Path) -> None:
    strengths = summary["strength"].to_numpy()
    asr = summary["safety_rate_pct"].to_numpy()
    accuracy = summary["capability_rate_pct"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(strengths, asr, "o-", label="HarmBehavior ASR (lower is safer)")
    axes[0].plot(strengths, accuracy, "s-", label="GSM8K accuracy")
    axes[0].set(xlabel="Controller strength s", ylabel="Percent", xticks=strengths)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(asr, accuracy, "o-")
    for strength, x, y in zip(strengths, asr, accuracy):
        axes[1].annotate(f"s={strength:g}", (x, y), xytext=(4, 4),
                         textcoords="offset points", fontsize=8)
    axes[1].set(xlabel="HarmBehavior ASR (%)", ylabel="GSM8K accuracy (%)")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Fixed top-25 neuron controller: safety–capability sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise ValueError("bootstrap-samples and permutation-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    safety_frame, safety = safety_data(args.safety_dir)
    capability_frames, capability = capability_data(args.capability_dir)
    strengths = sorted(set(safety) & set(capability))
    if strengths != sorted(safety) or strengths != sorted(capability):
        raise ValueError("Safety and capability strength grids differ")

    summary_rows = []
    test_rows = []
    safety_baseline = safety[0.0]
    capability_baseline = capability[0.0]
    for strength in strengths:
        safety_group = safety_frame[safety_frame["epsilon"] == strength]
        capability_group = capability_frames[strength]
        safety_count = int(safety[strength].sum())
        capability_count = int(capability[strength].sum())
        safety_ci = wilson_interval(safety_count, len(safety[strength]))
        capability_ci = wilson_interval(capability_count, len(capability[strength]))
        summary_rows.append({
            "strength": strength,
            "positive_scale": 1 + strength,
            "negative_scale": max(0.0, 1 - strength),
            "safety_n": len(safety[strength]),
            "safety_unsafe": safety_count,
            "safety_rate_pct": 100 * safety_count / len(safety[strength]),
            "safety_wilson_low_pct": safety_ci[0],
            "safety_wilson_high_pct": safety_ci[1],
            "safety_repetitive": int(safety_group["repetitive"].sum()),
            "safety_median_words": float(safety_group["word_count"].median()),
            "capability_n": len(capability[strength]),
            "capability_correct": capability_count,
            "capability_rate_pct": 100 * capability_count / len(capability[strength]),
            "capability_wilson_low_pct": capability_ci[0],
            "capability_wilson_high_pct": capability_ci[1],
            "capability_repetitive": int(capability_group["repetitive"].sum()),
            "capability_extraction_failures": int(capability_group["prediction"].isna().sum()),
            "capability_median_tokens": float(capability_group["response_token_count"].median()),
        })
        if strength == 0:
            continue
        safety_effect = paired_effect(
            safety_baseline, safety[strength], rng, args.bootstrap_samples
        )
        capability_effect = paired_effect(
            capability_baseline, capability[strength], rng, args.bootstrap_samples
        )
        test_rows.append({"benchmark": "HarmBehavior ASR", "strength": strength,
                          **safety_effect})
        test_rows.append({"benchmark": "GSM8K accuracy", "strength": strength,
                          **capability_effect})

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    tests["holm_p_within_benchmark"] = math.nan
    for benchmark, indices in tests.groupby("benchmark").groups.items():
        positions = list(indices)
        adjusted = holm_adjust(tests.loc[positions, "mcnemar_exact_p"].tolist())
        tests.loc[positions, "holm_p_within_benchmark"] = adjusted

    safety_rho, safety_p = exact_spearman(
        summary["strength"].tolist(), summary["safety_rate_pct"].tolist()
    )
    capability_rho, capability_p = exact_spearman(
        summary["strength"].tolist(), summary["capability_rate_pct"].tolist()
    )
    safety_gain = summary.loc[0, "safety_rate_pct"] - summary["safety_rate_pct"]
    capability_loss = summary.loc[0, "capability_rate_pct"] - summary["capability_rate_pct"]
    tradeoff_rho, tradeoff_p = exact_spearman(
        safety_gain.tolist(), capability_loss.tolist()
    )
    safety_matrix = np.column_stack([safety[strength] for strength in strengths])
    capability_matrix = np.column_stack([capability[strength] for strength in strengths])
    safety_q, safety_q_p = cochran_q(safety_matrix)
    capability_q, capability_q_p = cochran_q(capability_matrix)
    strength_array = np.asarray(strengths)
    safety_slope, safety_slope_p = paired_permutation_trend(
        strength_array, safety_matrix, rng, args.permutation_samples
    )
    capability_slope, capability_slope_p = paired_permutation_trend(
        strength_array, capability_matrix, rng, args.permutation_samples
    )
    trends = {
        "matched_binary_omnibus": {
            "safety_cochran_q": safety_q,
            "safety_p": safety_q_p,
            "capability_cochran_q": capability_q,
            "capability_p": capability_q_p,
            "degrees_of_freedom": len(strengths) - 1,
        },
        "paired_permutation_linear_trend": {
            "safety_slope_pp_per_strength": safety_slope,
            "safety_two_sided_p": safety_slope_p,
            "capability_slope_pp_per_strength": capability_slope,
            "capability_two_sided_p": capability_slope_p,
            "permutations": args.permutation_samples,
        },
        "strength_vs_safety_asr": {"spearman_rho": safety_rho,
                                    "exact_permutation_p": safety_p},
        "strength_vs_gsm8k_accuracy": {"spearman_rho": capability_rho,
                                        "exact_permutation_p": capability_p},
        "safety_gain_vs_capability_loss_across_strengths": {
            "spearman_rho": tradeoff_rho,
            "exact_permutation_p": tradeoff_p,
            "note": "Descriptive dose-level association; strength conditions are not independent samples.",
        },
    }

    summary.to_csv(args.output_dir / "tradeoff_summary.csv", index=False)
    tests.to_csv(args.output_dir / "paired_tests.csv", index=False)
    payload = {
        "design": {
            "controller": "fixed global top-25 neurons selected on disjoint AdvBench rows 100-299",
            "scaling": "positive = 1+s; negative = max(0, 1-s)",
            "safety": "same first 100 HarmBehavior examples at every strength",
            "capability": "same first 100 GSM8K test examples at every strength",
            "decoding": "greedy",
            "uncertainty": "Wilson 95% intervals for rates; paired bootstrap percentile 95% intervals for changes",
            "hypothesis_tests": "two-sided exact McNemar/binomial tests with Holm correction within each benchmark",
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "seed": args.seed,
        },
        "trends": trends,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_plot(summary, args.output_dir / "tradeoff_curve.png")
    print(summary.to_string(index=False), flush=True)
    print("\nPaired tests:\n" + tests.to_string(index=False), flush=True)
    print("\nTrends:\n" + json.dumps(trends, indent=2), flush=True)


if __name__ == "__main__":
    main()
