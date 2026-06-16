# Responsible AI Monitoring — UK Government AI Triage System

A Python monitoring package I built to track fairness, robustness, and transparency
for an AI system that triages citizen service requests into priority categories
(urgent / standard / low).

The system processes ~2,000 requests a day and has access to inference logs plus
a separate demographic dataset joined by request ID. The goal was to detect bias,
input drift, and data quality issues before they become governance problems.

---

## What's in here

```
├── technical_note.md    # design rationale, metric choices, thresholds, limitations
├── src/
│   ├── data_loader.py   # generates synthetic telemetry (inference logs + demographics)
│   ├── monitor.py       # runs the full pipeline and outputs a JSON report
│   └── metrics/
│       ├── fairness.py       # SPD, Disparate Impact, Confidence Disparity
│       ├── robustness.py     # PSI, KS test, prediction drift, confidence trend
│       └── transparency.py  # missingness rate, feature completeness
└── tests/
    └── test_metrics.py  # 19 unit tests
```

---

## Running it

```bash
pip install -r requirements.txt

# run the monitoring pipeline — outputs a structured JSON report
cd src
python monitor.py

# run tests
pytest tests/ -v
```

---

## Metrics

**Fairness** — evaluated separately for gender and location attributes.

- Statistical Parity Difference and Disparate Impact (four-fifths rule threshold from EEOC 1978)
- Reference group is fixed to the largest group by volume, not the highest-rate group —
  using the max-rate group as reference shifts daily and makes trend comparison meaningless
- Bootstrap 95% confidence intervals on all group-level rates (500 resamples)
- Groups with n < 50 get flagged as unreliable rather than computed — small-sample DI
  values without CIs aren't actionable

**Robustness** — two complementary drift methods because they catch different things.

- PSI detects mass shifts in the input distribution (stable/moderate/major thresholds
  from Siddiqi 2006 credit scoring literature)
- KS test catches shape changes that PSI can miss
- Prediction distribution shift flags sudden changes in urgent/standard/low proportions
- Confidence percentile trend (p5/p50/p95) as an early warning for model degradation

**Transparency** — data quality checks that validity of the above metrics depends on.

- Demographic matching rate: if 30% of records don't join, the fairness numbers are
  computed on a biased subset and shouldn't be trusted
- Feature completeness per input field

Alert thresholds are two-tier (warning / critical) and defined in a single config dict
in `monitor.py` so they can be tuned without touching computation logic.

---

## Design notes

The frequency decision came down to sample size. At 2,000 requests/day, running metrics
hourly gives ~83 requests per window — not enough for stable fairness or drift tests.
Daily batch gives sufficient power. For small demographic subgroups (under 50/day),
I use a rolling 7-day window instead.

One thing worth flagging explicitly: fairness metrics based on demographic parity and
equalised odds can't both be satisfied simultaneously when base rates differ across groups
(Chouldechova 2017). That's not a bug in the metrics — it's a fundamental constraint that
governance documentation needs to record up front.

Regulatory mapping for each metric (EU AI Act, NIST AI RMF, UK AI Playbook) is in
`technical_note.md`.

---

## References

- Hardt et al. (2016). Equality of opportunity in supervised learning. *NeurIPS 29*
- Chouldechova (2017). Fair prediction with disparate impact. *Big Data 5(2)*
- EEOC (1978). Uniform Guidelines on Employee Selection Procedures. 29 CFR 1607
- Guo et al. (2017). On calibration of modern neural networks. *ICML*
- Siddiqi (2006). *Credit Risk Scorecards*. Wiley
