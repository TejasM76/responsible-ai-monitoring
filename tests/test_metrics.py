import pandas as pd
import numpy as np

from src.metrics.fairness import compute_statistical_parity, compute_confidence_disparity, MIN_GROUP_SAMPLE
from src.metrics.robustness import (
    compute_prediction_drift,
    compute_feature_drift_ks,
    compute_psi,
    compute_confidence_trend,
)
from src.metrics.transparency import compute_demographic_missingness, compute_feature_completeness


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

def test_statistical_parity_basic():
    df = pd.DataFrame({
        "gender": ["M"] * 60 + ["F"] * 60,
        "confidence_score": [0.8] * 120,
        "predicted_priority": (
            ["urgent"] * 40 + ["standard"] * 20   # M: 40/60 = 0.667 urgent
            + ["urgent"] * 20 + ["standard"] * 40  # F: 20/60 = 0.333 urgent
        ),
    })
    res = compute_statistical_parity(df, "gender", positive_class="urgent")
    assert res["reference_group"] == "M"
    assert np.isclose(res["metrics"]["M"]["positive_rate"], 40 / 60)
    assert np.isclose(res["metrics"]["F"]["positive_rate"], 20 / 60)
    assert np.isclose(res["metrics"]["F"]["disparate_impact"], 0.5, atol=0.01)
    assert res["metrics"]["M"]["reliable"] is True
    assert res["metrics"]["F"]["reliable"] is True


def test_statistical_parity_reference_group_is_largest_not_highest_rate():
    # F is 100 rows (larger group) but M has higher urgent rate.
    # Reference should be F (largest), not M (highest rate).
    df = pd.DataFrame({
        "gender": ["M"] * 60 + ["F"] * 100,
        "confidence_score": [0.8] * 160,
        "predicted_priority": (
            ["urgent"] * 50 + ["standard"] * 10
            + ["urgent"] * 40 + ["standard"] * 60
        ),
    })
    res = compute_statistical_parity(df, "gender")
    assert res["reference_group"] == "F"


def test_statistical_parity_small_group_flagged():
    # A group with n < MIN_GROUP_SAMPLE should be flagged, not computed.
    small_n = MIN_GROUP_SAMPLE - 1
    df = pd.DataFrame({
        "gender": ["M"] * 60 + ["Nonbinary"] * small_n,
        "confidence_score": [0.8] * (60 + small_n),
        "predicted_priority": (
            ["urgent"] * 40 + ["standard"] * 20
            + ["urgent"] * (small_n // 2) + ["standard"] * (small_n - small_n // 2)
        ),
    })
    res = compute_statistical_parity(df, "gender")
    nonbinary = res["metrics"]["Nonbinary"]
    assert nonbinary["reliable"] is False
    assert nonbinary["positive_rate"] is None


def test_statistical_parity_bootstrap_ci_present():
    df = pd.DataFrame({
        "gender": ["M"] * 60 + ["F"] * 60,
        "confidence_score": [0.8] * 120,
        "predicted_priority": ["urgent"] * 60 + ["standard"] * 60,
    })
    res = compute_statistical_parity(df, "gender", n_bootstrap=100)
    ci = res["metrics"]["M"]["positive_rate_95ci"]
    assert "lower" in ci and "upper" in ci
    assert ci["lower"] <= res["metrics"]["M"]["positive_rate"] <= ci["upper"]


def test_statistical_parity_missing_attribute_returns_empty():
    df = pd.DataFrame({"predicted_priority": ["urgent", "standard"]})
    assert compute_statistical_parity(df, "gender") == {}


def test_confidence_disparity_delta_from_global():
    df = pd.DataFrame({
        "gender": ["M"] * 60 + ["F"] * 60,
        "confidence_score": [0.9] * 60 + [0.7] * 60,
        "predicted_priority": ["urgent"] * 120,
    })
    res = compute_confidence_disparity(df, "gender")
    global_med = res["global_median_confidence"]
    # Global median of [0.9]*60 + [0.7]*60 = 0.8
    assert np.isclose(global_med, 0.8, atol=0.01)
    assert res["groups"]["F"]["delta_from_global_median"] < 0
    assert res["groups"]["M"]["delta_from_global_median"] > 0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_prediction_drift_values():
    current_df = pd.DataFrame({"predicted_priority": ["urgent"] * 2 + ["low", "standard"]})
    baseline_df = pd.DataFrame({"predicted_priority": ["urgent", "standard", "low", "standard"]})
    drifts = compute_prediction_drift(current_df, baseline_df)
    assert np.isclose(drifts["urgent"], 0.25)
    assert np.isclose(drifts["low"], 0.0)
    assert np.isclose(drifts["standard"], 0.25)


def test_prediction_drift_no_drift():
    df = pd.DataFrame({"predicted_priority": ["urgent", "standard", "low"]})
    drifts = compute_prediction_drift(df, df.copy())
    for val in drifts.values():
        assert np.isclose(val, 0.0)


def test_ks_drift_identical_distributions():
    df = pd.DataFrame({"feature_x": np.random.normal(0, 1, 200)})
    result = compute_feature_drift_ks(df, df.copy(), "feature_x")
    assert result["ks_stat"] == 0.0
    assert result["p_value"] == 1.0


def test_ks_drift_very_different_distributions():
    current = pd.DataFrame({"feature_x": np.random.normal(10, 1, 300)})
    baseline = pd.DataFrame({"feature_x": np.random.normal(0, 1, 300)})
    result = compute_feature_drift_ks(current, baseline, "feature_x")
    assert result["ks_stat"] > 0.5
    assert result["p_value"] < 0.001


def test_ks_drift_missing_feature_returns_empty():
    df = pd.DataFrame({"other": [1, 2, 3]})
    assert compute_feature_drift_ks(df, df, "feature_x") == {}


def test_psi_stable_identical_distributions():
    df = pd.DataFrame({"feature_x": np.random.normal(0, 1, 500)})
    result = compute_psi(df, df.copy(), "feature_x")
    assert result["psi"] < 0.1
    assert result["interpretation"] == "stable"


def test_psi_major_shift():
    baseline = pd.DataFrame({"feature_x": np.random.normal(0, 1, 500)})
    current = pd.DataFrame({"feature_x": np.random.normal(5, 1, 500)})
    result = compute_psi(current, baseline, "feature_x")
    assert result["psi"] > 0.2
    assert result["interpretation"] == "major_shift"


def test_psi_missing_feature_returns_empty():
    df = pd.DataFrame({"other": [1, 2, 3]})
    assert compute_psi(df, df, "feature_x") == {}


def test_confidence_trend_percentiles():
    scores = list(range(1, 101))  # 1..100
    df = pd.DataFrame({"confidence_score": scores})
    result = compute_confidence_trend(df)
    assert result["p05"] < result["p50"] < result["p95"]
    assert np.isclose(result["mean"], 50.5)


# ---------------------------------------------------------------------------
# Transparency
# ---------------------------------------------------------------------------

def test_demographic_missingness_no_missing():
    df = pd.DataFrame({"gender": ["M", "F", "M"], "location": ["Urban", "Rural", "Urban"]})
    result = compute_demographic_missingness(df, ["gender", "location"])
    assert result["gender"] == 0.0
    assert result["location"] == 0.0


def test_demographic_missingness_partial():
    df = pd.DataFrame({"gender": ["M", None, "F", None, "M"]})
    result = compute_demographic_missingness(df, ["gender"])
    assert np.isclose(result["gender"], 0.4)


def test_demographic_missingness_empty_df():
    df = pd.DataFrame({"gender": pd.Series([], dtype=str)})
    result = compute_demographic_missingness(df, ["gender"])
    assert result == {}


def test_feature_completeness_full():
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    result = compute_feature_completeness(df, ["f1", "f2"])
    assert result["f1"] == 1.0
    assert result["f2"] == 1.0


def test_feature_completeness_with_nulls():
    df = pd.DataFrame({"f1": [1.0, None, 3.0, None]})
    result = compute_feature_completeness(df, ["f1"])
    assert np.isclose(result["f1"], 0.5)
