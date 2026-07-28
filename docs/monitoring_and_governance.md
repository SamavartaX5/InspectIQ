# Monitoring and governance

Day 6 compares the unlabelled 2023 candidate queue with the 2022 validation-feature population (primary chronological reference) and the 2020–2021 training-feature population (secondary stability reference). Targets are removed before drift calculations. The dashboard reads the saved report only.

Numeric PSI compares reference and current bin shares: it grows when records move between reference-defined quantile bins. Categorical PSI and Jensen–Shannon divergence compare category-share distributions, including an explicit missing category. Thresholds are configurable operational starting points, not universal statistical laws.

Score-distribution drift compares 2023 raw scores to frozen 2022 validation scores. It is not performance drift: no 2023 outcomes, Recall, Precision, Lift, AUC, Brier score, calibration, or fairness outcome metric is available.

Review-exposure diagnostics describe group shares in the candidate queue and review tiers. Small groups are suppressed for ratios. They do not identify protected attributes or establish discrimination; historical inspection selection and operational processes can create differences.

The review worksheet preserves rank and advisory model score while leaving human fields empty. Reviewers may record `pending`, `review_later`, `request_additional_information`, `escalate_for_human_assessment`, or `no_further_review`, provide a reason, and override priority. No enforcement action is automatic.

Future outcome evaluation requires a separate complete-label contract: `activity_nr`, `outcome_observed_date`, `serious_violation_found`, `label_source`, and `label_completeness_status`. It must join labels only after they are available, retain frozen predictions, exclude unknown labels, report label coverage, compare to the frozen baseline, and calculate Recall/Precision/Lift at 5/10/20%, PR-AUC, ROC-AUC, and Brier score only then.
