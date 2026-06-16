# Technical Note: Monitoring Approach for UK Government AI Triage System

**Context:** A UK government department triages ~2,000 citizen requests/day into priority
categories (urgent, standard, low) using an AI system. Available data: inference logs (input
metadata, predicted priority, confidence score, timestamp) joinable to demographic records by
request ID.

---

## 1. Metrics to Compute

### Fairness and Non-Discrimination

Because ground truth labels are unavailable at inference time, fairness is assessed via
demographic parity and confidence-based proxies.

**Statistical Parity Difference (SPD)** — absolute difference in urgent classification rate between
each demographic group and a fixed reference group (largest group by volume, to avoid
data-driven reference instability across daily windows). Formally:
SPD = P(Ŷ=urgent | G=g) − P(Ŷ=urgent | G=ref).
Reference: Hardt et al. (2016).

**Disparate Impact (DI)** — ratio of urgent rates between a group and the reference:
DI = P(Ŷ=urgent | G=g) / P(Ŷ=urgent | G=ref).
Threshold from the EEOC four-fifths rule (1978): DI ∈ [0.8, 1.25] is acceptable.

**Confidence Calibration Disparity** — median confidence score per demographic group with
bootstrap 95% confidence intervals (n=500 resamples). Systematically lower confidence for a
group suggests underrepresentation in training data (Guo et al., 2017). All group-level metrics
require n ≥ 50 observations per window; smaller subgroups are flagged as statistically
unreliable rather than silently computed.

**Fairness–Calibration Tension (flagged):** When ground truth becomes available (e.g., from
downstream case resolution), equalised odds (equal TPR and FPR across groups) should be
computed. Chouldechova (2017) proves that statistical parity, equalised odds, and calibration
cannot be simultaneously satisfied when base rates differ across groups — this trade-off must
be made explicit in governance documentation before deployment, not resolved post-hoc.

### Robustness and Distribution Shift

**Population Stability Index (PSI)** — bins continuous input features using the training baseline
distribution, then computes PSI = Σ (actual% − expected%) × ln(actual% / expected%).
Interpretation: PSI < 0.1 stable, 0.1–0.2 moderate shift, > 0.2 major shift (Siddiqi, 2006).
Applied to all continuous input features.

**KS Test** — Kolmogorov–Smirnov two-sample test as a complementary distributional shift
detector, particularly sensitive to changes in distribution shape rather than overall mass shifts.
KS statistic > 0.1 with p < 0.05 flags a significant shift.

**Prediction Distribution Shift** — absolute change in daily class proportions (urgent / standard /
low) relative to a 30-day rolling baseline. A sudden shift in urgent rate (e.g., 10% → 40%)
indicates model miscalibration or a genuine emergency event requiring human review.

**Confidence Score Trend** — 5th, 50th, and 95th percentiles of daily confidence scores with
7-day rolling comparison. Declining p50 or collapsing p5–p95 spread indicates degrading model
confidence on the current input population.

### Transparency and Auditability

**Demographic Matching Rate** — percentage of requests where demographic records are
successfully joined. Low match rates directly undermine the validity of all fairness metrics and
must be reported alongside them.

**Feature Completeness** — completion rate per input metadata field. Systematic missingness
in specific feature types may indicate upstream data pipeline failures.

---

## 2. Computation Frequency

**Daily batch** is the primary cadence. With ~2,000 requests/day, hourly computation (~83
requests/hour) produces statistically unstable fairness and drift metrics. A daily n=2,000
provides sufficient power (> 0.9) for KS and PSI tests at effect sizes of practical significance.

For small demographic subgroups (n < 50/day, e.g., non-binary individuals), a **rolling 7-day
window** accumulates sufficient sample size before metrics are computed. Metrics on windows
with n < 50 are suppressed and flagged rather than reported as reliable values.

---

## 3. Thresholds and Alerts

A two-tier system (Warning / Critical) avoids alert fatigue while ensuring escalation for
genuine violations. Warnings trigger internal metric review; Critical alerts trigger human
escalation per the department's AI governance protocol.

| Metric | Warning | Critical |
|---|---|---|
| Disparate Impact | Outside [0.85, 1.15] | Outside [0.8, 1.25] |
| Statistical Parity Diff | > 0.05 | > 0.10 |
| Confidence Disparity | Group median < global − 5% | Group median < global − 10% |
| PSI (input features) | > 0.10 | > 0.20 |
| KS Statistic | > 0.10 (p < 0.05) | > 0.20 (p < 0.01) |
| Prediction Drift (daily) | > 5% absolute change | > 10% absolute change |
| Demographic Missingness | > 5% | > 15% |

---

## 4. Limitations

1. **No ground truth at inference time.** All fairness metrics are proxy-based (demographic
parity). Performance-based fairness — equalised odds, false negative rate parity — cannot be
computed until case outcomes are known. False negatives on urgent cases carry greater harm
than false positives; this asymmetry must be documented as a residual risk and reviewed when
outcome data becomes available.

2. **Small subgroup instability.** With 2,000 requests/day, intersectional groups (e.g.,
non-binary individuals from rural areas) may yield n < 30, making estimates unreliable even
with bootstrap correction. Bayesian hierarchical models (partial pooling) should be considered
for sustained monitoring of small subgroups.

3. **Selection bias in demographic data.** Voluntarily or administratively captured demographic
data may systematically under-represent marginalised groups, introducing unmeasurable
coverage bias into fairness assessments. The demographic matching rate metric partially
surfaces this but cannot correct for it.

4. **Simpson's Paradox.** A fairness disparity in aggregate may disappear or reverse when
conditioned on legitimate confounders (e.g., a geographic area experiencing a genuine
emergency, naturally elevating urgent rates for that demographic). All triggered alerts should
prompt confounding analysis before escalation.

5. **Fairness impossibility (Chouldechova, 2017).** When base rates differ across groups,
statistical parity and equalised odds cannot both be satisfied. Governance documentation must
record which fairness criterion is prioritised, why, and who is accountable for the decision.

---

## Regulatory Mapping

| Metric | EU AI Act | NIST AI RMF | UK AI Playbook |
|---|---|---|---|
| Statistical Parity / DI | Art. 10(2)(f) — non-discriminatory training data | Measure 2.5 — bias and fairness evaluation | Principle 1 — Fairness and non-discrimination |
| Confidence Calibration Disparity | Art. 15(1) — accuracy and robustness | Measure 2.6 — AI system performance monitoring | Principle 5 — Safety and robustness |
| PSI / KS Feature Drift | Art. 9(4) — risk management system; ongoing monitoring | Manage 4.1 — monitoring and corrective action | Principle 9 — Accountability and governance |
| Prediction Distribution Shift | Art. 15(3) — performance monitoring over lifetime | Measure 2.6 | Principle 5 — Safety and robustness |
| Demographic Missingness | Art. 10(3) — data governance and completeness | Map 5.1 — data quality assessment | Principle 6 — Transparency and explainability |
| Feature Completeness | Art. 10(2)(b) — data completeness requirements | Map 5.1 | Principle 6 — Transparency and explainability |

---

**References**

- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS 29*.
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data, 5*(2), 153–163.
- EEOC (1978). Uniform Guidelines on Employee Selection Procedures. 29 CFR Part 1607.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. John Wiley & Sons.
