import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import ks_2samp


def compute_psi(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    feature: str,
    bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute Population Stability Index (PSI) for a numerical feature.

    Bins are derived from the baseline distribution so the reference stays fixed
    across monitoring windows. PSI < 0.1 is stable, 0.1–0.2 is moderate shift,
    > 0.2 is major shift.

    Reference: Siddiqi (2006), Credit Risk Scorecards.
    """
    if feature not in current_df.columns or feature not in baseline_df.columns:
        return {}

    baseline_vals = baseline_df[feature].dropna().values
    current_vals = current_df[feature].dropna().values

    if len(baseline_vals) == 0 or len(current_vals) == 0:
        return {}

    # Bin edges fixed on baseline percentiles; np.unique removes duplicate edges
    bin_edges = np.unique(np.percentile(baseline_vals, np.linspace(0, 100, bins + 1)))
    if len(bin_edges) < 2:
        return {"psi": 0.0, "interpretation": "insufficient_unique_values"}

    baseline_counts, _ = np.histogram(baseline_vals, bins=bin_edges)
    current_counts, _ = np.histogram(current_vals, bins=bin_edges)

    # Small epsilon prevents log(0) and division by zero
    eps = 1e-8
    baseline_pct = baseline_counts / (len(baseline_vals) + eps)
    current_pct = current_counts / (len(current_vals) + eps)

    psi_value = float(
        np.sum((current_pct - baseline_pct) * np.log((current_pct + eps) / (baseline_pct + eps)))
    )

    if psi_value < 0.1:
        interpretation = "stable"
    elif psi_value < 0.2:
        interpretation = "moderate_shift"
    else:
        interpretation = "major_shift"

    return {"psi": psi_value, "interpretation": interpretation, "bin_count": len(bin_edges) - 1}


def compute_feature_drift_ks(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    feature: str,
) -> Dict[str, float]:
    """
    Kolmogorov-Smirnov two-sample test for distributional shift in a numerical feature.
    Complementary to PSI: KS is sensitive to shape changes, PSI to mass shifts.
    """
    if feature not in current_df.columns or feature not in baseline_df.columns:
        return {}

    stat, p_value = ks_2samp(
        current_df[feature].dropna().values,
        baseline_df[feature].dropna().values,
    )
    return {"ks_stat": float(stat), "p_value": float(p_value)}


def compute_prediction_drift(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Absolute change in predicted class proportions between current and baseline windows.
    """
    current_dist = current_df["predicted_priority"].value_counts(normalize=True).to_dict()
    baseline_dist = baseline_df["predicted_priority"].value_counts(normalize=True).to_dict()

    drifts: Dict[str, float] = {}
    for cls in set(current_dist.keys()).union(baseline_dist.keys()):
        drifts[cls] = abs(current_dist.get(cls, 0.0) - baseline_dist.get(cls, 0.0))
    return drifts


def compute_confidence_trend(df: pd.DataFrame) -> Dict[str, float]:
    """
    Percentile summary of confidence scores. Declining p50 or collapsing spread
    signals degrading model confidence on the current input population.
    """
    scores = df["confidence_score"]
    return {
        "p05": float(scores.quantile(0.05)),
        "p50": float(scores.median()),
        "p95": float(scores.quantile(0.95)),
        "mean": float(scores.mean()),
    }
