# RAI Tracker — ML Scientist Technical Assessment

A production-quality Python monitoring framework for a UK government AI triage system,
built as the technical assessment for the Machine Learning Scientist role at
[RAI Tracker (RAIT)](https://www.raitracker.com).

## The Scenario

A UK government department deploys an AI system that triages ~2,000 incoming citizen
requests per day into priority categories: `urgent`, `standard`, `low`. The system has
access to inference logs (input metadata, predicted priority, confidence score, timestamp)
and a separate demographic dataset joinable by request ID.

**Task:** Design and implement a monitoring approach covering fairness, robustness, and
transparency — with metrics, computation frequency, alert thresholds, and limitations.

---

## Project Structure

```
├── technical_note.md          # Written answer: metrics, frequency, thresholds, limitations
│                              # Includes regulatory mapping (EU AI Act, NIST AI RMF, UK AI Playbook)
│                              # and academic citations for every metric choice
├── requirements.txt
└── src/
    ├── data_loader.py         # Synthetic telemetry generation (inference logs + demographics)
    ├── monitor.py             # Main pipeline: join → compute → alert → JSON report
    └── metrics/
        ├── fairness.py        # Statistical Parity, Disparate Impact, Confidence Disparity
        │                      # with bootstrap 95% CIs and minimum sample size guards
        ├── robustness.py      # PSI, KS test, Prediction Drift, Confidence Trend
        └── transparency.py    # Demographic Missingness Rate, Feature Completeness
```

---

## Metrics Implemented

### Fairness (per protected attribute: gender, location)
| Metric | Method | Reference |
|---|---|---|
| Statistical Parity Difference | Difference in urgent classification rates vs reference group | Hardt et al. (2016) |
| Disparate Impact | Rate ratio; four-fifths rule threshold | EEOC (1978) |
| Confidence Calibration Disparity | Median confidence per group with bootstrap 95% CI | Guo et al. (2017) |

Reference group = largest group by volume (stable across daily windows).
Groups with n < 50 are flagged as statistically unreliable rather than silently computed.

### Robustness
| Metric | Method |
|---|---|
| Population Stability Index (PSI) | Percentile-binned baseline; PSI < 0.1 stable, > 0.2 major shift |
| KS Test | Two-sample Kolmogorov–Smirnov for shape changes in input features |
| Prediction Distribution Shift | Absolute change in class proportions vs baseline |
| Confidence Score Trend | p5 / p50 / p95 percentiles of daily confidence scores |

### Transparency
| Metric | What it detects |
|---|---|
| Demographic Matching Rate | % of requests where demographic join succeeds |
| Feature Completeness | % of input metadata fields with valid values |

---

## Alert Thresholds

Two-tier system (Warning / Critical) to avoid alert fatigue:

| Metric | Warning | Critical |
|---|---|---|
| Disparate Impact | Outside [0.85, 1.15] | Outside [0.8, 1.25] |
| Statistical Parity Diff | > 0.05 | > 0.10 |
| Confidence Disparity | Group median < global − 5% | Group median < global − 10% |
| PSI | > 0.10 | > 0.20 |
| Prediction Drift | > 5% | > 10% |
| Demographic Missingness | > 5% | > 15% |

---

## Regulatory Mapping

Each metric traces to specific requirements in:
- **EU AI Act** — Articles 9, 10, 13, 15
- **NIST AI RMF** — Measure 2.5, Measure 2.6, Map 5.1, Manage 4.1
- **UK AI Playbook** — Principles 1, 5, 6, 9

See [`technical_note.md`](technical_note.md) for the full mapping table.

---

## Setup

```bash
pip install -r requirements.txt
```

**Run the monitoring pipeline:**
```bash
cd src
python monitor.py
```

Outputs a structured JSON report with all metric values and triggered alerts.

**Run tests (19 unit tests):**
```bash
pytest tests/ -v
```

---

## Design Decisions

- **Daily batch over real-time:** ~83 requests/hour is too small for stable fairness metrics.
  Daily n=2,000 gives sufficient statistical power for KS and PSI tests.
- **Rolling 7-day window for small subgroups:** groups with n < 50/day accumulate samples
  before metrics are computed.
- **Bootstrap CIs on all fairness metrics:** point estimates without uncertainty bounds are
  not actionable. 500-resample bootstrap with fixed seed (42) for reproducibility.
- **Structured JSON logging:** required for CloudWatch / log aggregation in AWS Lambda context.
- **Thresholds externalised to config dict:** tunable without touching computation logic.
- **Fairness–calibration impossibility acknowledged:** Chouldechova (2017) proves statistical
  parity and equalised odds cannot both hold when base rates differ. Governance documentation
  must record which criterion is prioritised.

---

## References

- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS 29*.
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data, 5*(2), 153–163.
- EEOC (1978). Uniform Guidelines on Employee Selection Procedures. 29 CFR Part 1607.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. John Wiley & Sons.
