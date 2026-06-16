import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from data_loader import generate_mock_data, load_and_join_data
from metrics.fairness import compute_statistical_parity, compute_confidence_disparity
from metrics.robustness import (
    compute_psi,
    compute_prediction_drift,
    compute_feature_drift_ks,
    compute_confidence_trend,
)
from metrics.transparency import compute_demographic_missingness, compute_feature_completeness

# ---------------------------------------------------------------------------
# Structured JSON logging — required for CloudWatch/observability pipelines.
# Plain-text logging is not parseable by log aggregation tools.
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        })

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JsonFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# Alert thresholds — externalised from logic so they can be tuned without
# touching computation code. In production these come from a config store.
# ---------------------------------------------------------------------------
THRESHOLDS: Dict[str, Any] = {
    "fairness": {
        "disparate_impact_warn_low": 0.85,
        "disparate_impact_warn_high": 1.15,
        "disparate_impact_crit_low": 0.80,
        "disparate_impact_crit_high": 1.25,
        "spd_warn": 0.05,
        "spd_crit": 0.10,
        "confidence_disparity_warn_pct": 0.05,
        "confidence_disparity_crit_pct": 0.10,
    },
    "robustness": {
        "psi_warn": 0.10,
        "psi_crit": 0.20,
        "ks_stat_warn": 0.10,
        "ks_stat_crit": 0.20,
        "prediction_drift_warn": 0.05,
        "prediction_drift_crit": 0.10,
    },
    "transparency": {
        "missingness_warn": 0.05,
        "missingness_crit": 0.15,
    },
}

PROTECTED_ATTRIBUTES = ["gender", "location"]
NUMERICAL_FEATURES = ["input_feature_1", "input_feature_2"]


def _check_fairness_alerts(
    parity: Dict[str, Any],
    conf_disp: Dict[str, Any],
    alerts: List[Dict[str, str]],
    attribute: str,
) -> None:
    t = THRESHOLDS["fairness"]
    global_median = conf_disp.get("global_median_confidence")

    for group, m in parity.get("metrics", {}).items():
        if not m.get("reliable"):
            continue

        di = m["disparate_impact"]
        spd = m["statistical_parity_diff"]

        if di is not None:
            if di < t["disparate_impact_crit_low"] or di > t["disparate_impact_crit_high"]:
                level = "CRITICAL"
            elif di < t["disparate_impact_warn_low"] or di > t["disparate_impact_warn_high"]:
                level = "WARNING"
            else:
                level = None
            if level:
                alerts.append({
                    "level": level, "dimension": "fairness",
                    "message": (
                        f"Disparate Impact for {attribute}={group} is {di:.3f} "
                        f"(ref={parity['reference_group']})"
                    ),
                })

        if abs(spd) > t["spd_crit"]:
            alerts.append({
                "level": "CRITICAL", "dimension": "fairness",
                "message": f"SPD for {attribute}={group} is {spd:.3f}",
            })
        elif abs(spd) > t["spd_warn"]:
            alerts.append({
                "level": "WARNING", "dimension": "fairness",
                "message": f"SPD for {attribute}={group} is {spd:.3f}",
            })

    if global_median:
        for group, g in conf_disp.get("groups", {}).items():
            if not g.get("reliable"):
                continue
            delta = g["delta_from_global_median"]
            if delta < -t["confidence_disparity_crit_pct"]:
                alerts.append({
                    "level": "CRITICAL", "dimension": "fairness",
                    "message": (
                        f"Confidence disparity for {attribute}={group}: "
                        f"median={g['median_confidence']:.3f}, "
                        f"delta={delta:.3f} from global"
                    ),
                })
            elif delta < -t["confidence_disparity_warn_pct"]:
                alerts.append({
                    "level": "WARNING", "dimension": "fairness",
                    "message": (
                        f"Confidence disparity for {attribute}={group}: "
                        f"median={g['median_confidence']:.3f}, "
                        f"delta={delta:.3f} from global"
                    ),
                })


def _check_robustness_alerts(
    psi_results: Dict[str, Any],
    ks_results: Dict[str, Any],
    pred_drift: Dict[str, float],
    alerts: List[Dict[str, str]],
) -> None:
    t = THRESHOLDS["robustness"]

    for feature, result in psi_results.items():
        psi_val = result.get("psi", 0.0)
        if psi_val > t["psi_crit"]:
            alerts.append({
                "level": "CRITICAL", "dimension": "robustness",
                "message": f"PSI for {feature} is {psi_val:.3f} ({result['interpretation']})",
            })
        elif psi_val > t["psi_warn"]:
            alerts.append({
                "level": "WARNING", "dimension": "robustness",
                "message": f"PSI for {feature} is {psi_val:.3f} ({result['interpretation']})",
            })

    for feature, result in ks_results.items():
        ks_stat = result.get("ks_stat", 0.0)
        p_val = result.get("p_value", 1.0)
        if ks_stat > t["ks_stat_crit"] and p_val < 0.01:
            alerts.append({
                "level": "CRITICAL", "dimension": "robustness",
                "message": f"KS drift for {feature}: stat={ks_stat:.3f}, p={p_val:.4f}",
            })
        elif ks_stat > t["ks_stat_warn"] and p_val < 0.05:
            alerts.append({
                "level": "WARNING", "dimension": "robustness",
                "message": f"KS drift for {feature}: stat={ks_stat:.3f}, p={p_val:.4f}",
            })

    for cls, drift in pred_drift.items():
        if drift > t["prediction_drift_crit"]:
            alerts.append({
                "level": "CRITICAL", "dimension": "robustness",
                "message": f"Prediction drift for class '{cls}': {drift * 100:.1f}% absolute change",
            })
        elif drift > t["prediction_drift_warn"]:
            alerts.append({
                "level": "WARNING", "dimension": "robustness",
                "message": f"Prediction drift for class '{cls}': {drift * 100:.1f}% absolute change",
            })


def _check_transparency_alerts(
    missingness: Dict[str, float],
    alerts: List[Dict[str, str]],
) -> None:
    t = THRESHOLDS["transparency"]
    for col, rate in missingness.items():
        if rate > t["missingness_crit"]:
            alerts.append({
                "level": "CRITICAL", "dimension": "transparency",
                "message": f"Demographic missingness for '{col}': {rate * 100:.1f}%",
            })
        elif rate > t["missingness_warn"]:
            alerts.append({
                "level": "WARNING", "dimension": "transparency",
                "message": f"Demographic missingness for '{col}': {rate * 100:.1f}%",
            })


def run_pipeline() -> Dict[str, Any]:
    run_ts = datetime.now(timezone.utc).isoformat()
    logger.info("Generating mock data")
    current_inf, current_demo = generate_mock_data(num_requests=2000)
    baseline_inf, _ = generate_mock_data(num_requests=2000)

    logger.info("Joining inference logs with demographic data")
    df = load_and_join_data(current_inf, current_demo)

    alerts: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Fairness — evaluated per protected attribute
    # ------------------------------------------------------------------
    logger.info("Computing fairness metrics")
    fairness_results: Dict[str, Any] = {}
    for attr in PROTECTED_ATTRIBUTES:
        parity = compute_statistical_parity(df, protected_attribute=attr)
        conf_disp = compute_confidence_disparity(df, protected_attribute=attr)
        fairness_results[attr] = {
            "statistical_parity": parity,
            "confidence_disparity": conf_disp,
        }
        _check_fairness_alerts(parity, conf_disp, alerts, attr)

    # ------------------------------------------------------------------
    # Robustness — PSI + KS per feature, prediction drift, confidence trend
    # ------------------------------------------------------------------
    logger.info("Computing robustness metrics")
    psi_results = {f: compute_psi(current_inf, baseline_inf, f) for f in NUMERICAL_FEATURES}
    ks_results = {f: compute_feature_drift_ks(current_inf, baseline_inf, f) for f in NUMERICAL_FEATURES}
    pred_drift = compute_prediction_drift(current_inf, baseline_inf)
    conf_trend = compute_confidence_trend(df)

    _check_robustness_alerts(psi_results, ks_results, pred_drift, alerts)

    # ------------------------------------------------------------------
    # Transparency — missingness and feature completeness
    # ------------------------------------------------------------------
    logger.info("Computing transparency metrics")
    missingness = compute_demographic_missingness(df, PROTECTED_ATTRIBUTES)
    completeness = compute_feature_completeness(df, NUMERICAL_FEATURES)

    _check_transparency_alerts(missingness, alerts)

    # ------------------------------------------------------------------
    # Assemble report
    # ------------------------------------------------------------------
    report = {
        "run_timestamp": run_ts,
        "request_count": len(current_inf),
        "matched_demographic_count": len(df),
        "alerts": alerts,
        "fairness": fairness_results,
        "robustness": {
            "psi": psi_results,
            "ks_drift": ks_results,
            "prediction_drift": pred_drift,
            "confidence_trend": conf_trend,
        },
        "transparency": {
            "demographic_missingness": missingness,
            "feature_completeness": completeness,
        },
    }

    print(json.dumps(report, indent=2))
    logger.info(f"Pipeline complete — {len(alerts)} alert(s) triggered")
    return report


if __name__ == "__main__":
    run_pipeline()
