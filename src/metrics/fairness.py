import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Minimum observations per group before a metric is considered statistically reliable.
# With n < 50, bootstrap CIs are too wide to act on; smaller groups are flagged, not silenced.
MIN_GROUP_SAMPLE = 50


def _bootstrap_ci(
    values: np.ndarray,
    statistic: callable,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for an arbitrary statistic on a 1-D array."""
    rng = np.random.default_rng(seed=42)
    stats = [
        statistic(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    return (
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    )


def compute_statistical_parity(
    df: pd.DataFrame,
    protected_attribute: str,
    positive_class: str = "urgent",
    reference_group: Optional[str] = None,
    n_bootstrap: int = 500,
) -> Dict[str, Any]:
    """
    Statistical Parity Difference and Disparate Impact with bootstrap 95% CIs.

    The reference group defaults to the largest group by volume. Using the
    highest-rate group as reference (previous behaviour) is data-driven and can
    shift across daily windows, making trend comparisons unstable.

    Groups with n < MIN_GROUP_SAMPLE are reported as unreliable rather than
    silently computed — small-sample DI values are meaningless without CIs, and
    CIs on n < 50 binomial proportions span nearly [0, 1].

    References:
      Hardt et al. (2016), Equality of Opportunity in Supervised Learning, NeurIPS.
      EEOC (1978), Uniform Guidelines on Employee Selection Procedures, 29 CFR 1607.
    """
    if protected_attribute not in df.columns:
        return {}

    valid_df = df.dropna(subset=[protected_attribute])

    # Stable reference: largest group, or caller-specified
    if reference_group is None:
        reference_group = valid_df[protected_attribute].value_counts().idxmax()

    rates: Dict[str, Optional[float]] = {}
    cis: Dict[str, Tuple[float, float]] = {}
    counts: Dict[str, int] = {}

    for group in valid_df[protected_attribute].unique():
        group_df = valid_df[valid_df[protected_attribute] == group]
        n = len(group_df)
        counts[group] = n

        if n < MIN_GROUP_SAMPLE:
            rates[group] = None
            continue

        arr = (group_df["predicted_priority"] == positive_class).astype(float).values
        rates[group] = float(arr.mean())
        cis[group] = _bootstrap_ci(arr, np.mean, n_bootstrap=n_bootstrap)

    ref_rate = rates.get(reference_group)
    if ref_rate is None:
        return {
            "error": (
                f"Reference group '{reference_group}' has insufficient sample "
                f"size (n={counts.get(reference_group, 0)}, required >= {MIN_GROUP_SAMPLE})"
            )
        }

    results: Dict[str, Any] = {}
    for group, rate in rates.items():
        if rate is None:
            results[group] = {
                "positive_rate": None,
                "disparate_impact": None,
                "statistical_parity_diff": None,
                "count": counts[group],
                "reliable": False,
                "warning": (
                    f"Insufficient sample size (n={counts[group]}, "
                    f"required >= {MIN_GROUP_SAMPLE})"
                ),
            }
        else:
            di = rate / ref_rate if ref_rate > 0 else None
            results[group] = {
                "positive_rate": rate,
                "positive_rate_95ci": {"lower": cis[group][0], "upper": cis[group][1]},
                "disparate_impact": di,
                "statistical_parity_diff": ref_rate - rate,
                "count": counts[group],
                "reliable": True,
            }

    return {
        "reference_group": reference_group,
        "positive_class": positive_class,
        "min_group_sample_required": MIN_GROUP_SAMPLE,
        "metrics": results,
    }


def compute_confidence_disparity(
    df: pd.DataFrame,
    protected_attribute: str,
    n_bootstrap: int = 500,
) -> Dict[str, Any]:
    """
    Median confidence score per demographic group with bootstrap 95% CIs.

    Lower median confidence for a group relative to the global median indicates
    the model is less certain about predictions for that group — a proxy for
    underrepresentation in training data (Guo et al., 2017, ICML).
    """
    if protected_attribute not in df.columns:
        return {}

    valid_df = df.dropna(subset=[protected_attribute])
    global_median = float(valid_df["confidence_score"].median())
    results: Dict[str, Any] = {}

    for group in valid_df[protected_attribute].unique():
        group_df = valid_df[valid_df[protected_attribute] == group]
        n = len(group_df)

        if n < MIN_GROUP_SAMPLE:
            results[group] = {
                "median_confidence": None,
                "count": n,
                "reliable": False,
                "warning": f"Insufficient sample size (n={n}, required >= {MIN_GROUP_SAMPLE})",
            }
            continue

        arr = group_df["confidence_score"].values
        median = float(np.median(arr))
        lower, upper = _bootstrap_ci(arr, np.median, n_bootstrap=n_bootstrap)
        results[group] = {
            "median_confidence": median,
            "median_confidence_95ci": {"lower": lower, "upper": upper},
            "delta_from_global_median": round(median - global_median, 4),
            "count": n,
            "reliable": True,
        }

    return {"global_median_confidence": global_median, "groups": results}
