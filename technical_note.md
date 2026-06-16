# Technical Note: Monitoring Approach for UK Government AI Triage System

The system triages ~2,000 citizen requests a day into urgent, standard, and low priority
categories. Monitoring data is two separate sources — inference logs (input metadata,
predicted priority, confidence score, timestamp) and a demographic dataset joinable by
request ID. The challenge is designing monitoring that works without ground truth labels,
which won't be available at inference time.

---

## Metrics

### Fairness

Without outcome data, fairness has to be assessed through demographic parity and
confidence-based proxies. That's a real limitation and I flag it explicitly below, but
it's the best available option given what the system exposes.

**Statistical Parity Difference (SPD)** measures how much the urgent classification rate
differs between demographic groups. Formally: P(Ŷ=urgent | G=g) − P(Ŷ=urgent | G=ref).
The reference group is fixed to the largest group by volume — not the highest-rate group,
which would shift daily and make trend comparisons unstable across monitoring windows.
(Hardt et al., 2016)

**Disparate Impact (DI)** is the ratio version of the same thing: rate for group g divided
by rate for the reference group. The four-fifths rule (EEOC, 1978) flags DI outside
[0.8, 1.25] as a discrimination indicator. This threshold comes from employment law but
has become the standard starting point for algorithmic fairness audits.

Both metrics are reported with bootstrap 95% confidence intervals (500 resamples, seeded
for reproducibility). A DI of 0.79 without a CI is not actionable — you don't know if
that's signal or noise. Groups with fewer than 50 observations in a window are flagged as
statistically unreliable and excluded from alert evaluation.

**Confidence Calibration Disparity** — the median model confidence score per demographic
group compared to the global median. Systematically lower confidence for a group suggests
underrepresentation in training data (Guo et al., 2017). It's a weak proxy but it's
observable without labels and it's early-warning signal worth tracking.

One important thing to note upfront: Chouldechova (2017) proves that when base rates
differ across groups, you cannot simultaneously achieve statistical parity, equalised odds,
and calibration. This isn't a bug in the metrics — it's a fundamental constraint.
Governance documentation needs to record which fairness criterion is being prioritised and
why, before deployment, not after a violation is found.

When outcome data eventually becomes available (e.g., from case resolution records),
equalised odds should be computed — equal true positive and false positive rates across
groups. For a triage system, false negatives on urgent cases carry more harm than false
positives, so FNR parity deserves particular attention.

### Robustness

**Population Stability Index (PSI)** — bins input features using the training baseline
distribution and measures how much today's distribution has shifted. PSI = Σ (actual% −
expected%) × ln(actual% / expected%). Standard interpretation from credit scoring
literature (Siddiqi, 2006): below 0.1 is stable, 0.1–0.2 is worth investigating,
above 0.2 is a significant shift requiring review.

**KS test (Kolmogorov–Smirnov)** runs alongside PSI because they catch different things.
PSI is sensitive to mass shifts across bins; KS is sensitive to shape changes in the
distribution. Using both reduces the chance of missing a real drift event.

**Prediction distribution shift** tracks the day-over-day change in how many requests
land in each priority class. A sudden jump in urgent rate from 10% to 40% either means
the model is broken or a genuine emergency is happening — both require human review.

**Confidence score trend** (p5/p50/p95 percentiles) gives an early warning of model
degradation. A steady decline in median confidence, or a collapse of the p5–p95 spread,
indicates the model is less certain about the current input population than it used to be.

### Transparency

**Demographic matching rate** — the percentage of requests where demographic records
successfully join. This one matters more than it looks. If 30% of records don't join, the
fairness metrics above are computed on a subset that may be systematically different from
the full population. The matching rate needs to be reported alongside fairness numbers,
not buried in a separate data quality report.

**Feature completeness** — completion rate per input metadata field. Systematic missingness
in specific fields can indicate upstream pipeline failures rather than random data loss.

---

## Computation Frequency

Daily batch. At 2,000 requests/day, hourly computation gives ~83 requests per window —
not enough for KS or PSI tests to have meaningful statistical power, and fairness metrics
on 83 requests will have CIs so wide they're useless. Daily n=2,000 is sufficient.

For small demographic subgroups (under 50 observations per day, e.g., non-binary
individuals), a rolling 7-day window is used instead of daily. Metrics on windows that
still don't reach 50 are suppressed with a flag rather than reported.

---

## Thresholds and Alerts

Two-tier system to avoid alert fatigue. Warnings trigger internal review; Critical alerts
trigger escalation to the governance team.

| Metric | Warning | Critical |
|---|---|---|
| Disparate Impact | Outside [0.85, 1.15] | Outside [0.8, 1.25] |
| Statistical Parity Diff | > 0.05 | > 0.10 |
| Confidence Disparity | Group median < global − 5% | Group median < global − 10% |
| PSI | > 0.10 | > 0.20 |
| KS Statistic | > 0.10 (p < 0.05) | > 0.20 (p < 0.01) |
| Prediction Drift | > 5% absolute change | > 10% absolute change |
| Demographic Missingness | > 5% | > 15% |

These starting thresholds are based on published conventions (EEOC four-fifths rule, PSI
from Siddiqi). They should be calibrated to the system's specific risk tolerance and
base rates once initial monitoring data is available.

---

## Limitations

**No ground truth at inference time.** This is the biggest constraint. Every fairness
metric here is a proxy. I can tell you that Group A gets urgent classification 20% less
often than Group B, but I can't tell you whether that's unfair or accurate without knowing
the true priority of those requests. When outcome data becomes available, the monitoring
approach needs to add performance-based fairness metrics (FNR parity, equalised odds).

**Small subgroup instability.** With 2,000 requests/day, any intersectional slice
(e.g., non-binary individuals from rural areas) may have 20–40 observations. Bootstrap
CIs on n=30 binomial proportions span nearly the full [0,1] range. The n ≥ 50 guard
handles this for daily windows, but for sustained small-population monitoring, Bayesian
hierarchical models with partial pooling would be more appropriate than frequentist
point estimates with wide CIs.

**Selection bias in demographic data.** If demographic data is voluntarily provided,
the people who provide it may be systematically different from those who don't. The
demographic matching rate metric surfaces the extent of this problem but can't correct for
it. Any fairness report needs to caveat this explicitly.

**Simpson's Paradox.** A group-level disparity in aggregate might disappear or reverse
when you condition on legitimate confounders. A specific geographic area experiencing a
genuine emergency will have a higher urgent rate for the demographics concentrated there.
All triggered alerts should prompt confounding analysis before escalation.

**Fairness impossibility.** As noted above (Chouldechova, 2017) — can't simultaneously
satisfy statistical parity, equalised odds, and calibration when base rates differ.
This is a governance decision, not a technical one.

---

## Regulatory Mapping

| Metric | EU AI Act | NIST AI RMF | UK AI Playbook |
|---|---|---|---|
| Statistical Parity / DI | Art. 10(2)(f) — non-discriminatory data | Measure 2.5 — bias evaluation | Principle 1 — Fairness |
| Confidence Calibration Disparity | Art. 15(1) — accuracy and robustness | Measure 2.6 — performance monitoring | Principle 5 — Safety |
| PSI / KS Drift | Art. 9(4) — risk management, ongoing monitoring | Manage 4.1 — corrective action | Principle 9 — Accountability |
| Prediction Distribution Shift | Art. 15(3) — lifetime performance monitoring | Measure 2.6 | Principle 5 |
| Demographic Missingness | Art. 10(3) — data governance | Map 5.1 — data quality | Principle 6 — Transparency |

---

**References**

- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS 29*
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data, 5*(2), 153–163
- EEOC (1978). Uniform Guidelines on Employee Selection Procedures. 29 CFR Part 1607
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley
