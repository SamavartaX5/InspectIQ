# Model card: InspectIQ selected candidate

## Model summary

The selected candidate is `exp_05_random_forest`, a Random Forest trained on historical California OSHA inspection data. The final package uses an **uncalibrated model output** for ranking supplied candidates.

## Intended use

Prioritize a fixed human-review budget within a supplied candidate batch. It supports retrospective evidence review and should be paired with reviewer rationale, override, and escalation controls.

## Out-of-scope uses

It is not a violation finding, calibrated probability, workplace-risk registry, enforcement target generator, or autonomous inspection/enforcement mechanism.

## Data, labels, and periods

- Training data: 2020–2021, 1,200 labelled rows.
- Validation data: 2022, 600 labelled rows (154 positive, 446 negative).
- Locked candidate data: 2023, 300 rows with no target in the workflow.
- Label: at least one non-deleted Serious, Willful, or Repeat violation joined by `activity_nr`.
- Unknown/incomplete retrieval: excluded, never converted into a negative.

Historical OSHA data is selection-biased. No protected demographic attributes are available for outcome-fairness evaluation.

## Features and leakage controls

Feature groups include inspection fields (`naics_group`, inspection type/scope, owner type, safety/health), establishment-size proxy, open month, and strictly-prior industry history. Establishment-history features are omitted because no defensible establishment identifier is available. Feature version `day2-historical-v1` prohibits own-row, same-day, future, validation-to-validation, and candidate feedback leakage.

## Baseline and candidate models

The baseline is a smoothed, training-only industry-rate ranking with explicit fallbacks. Eight candidates were compared. The selected Random Forest has `n_estimators=200`, `max_depth=8`, `min_samples_leaf=5`, `class_weight=None`, and `random_state=42`.

## Validation results

On the labelled 2022 validation period, the selected model captured 36 positives in the top 10% (60 rows), with Precision@10%=0.6000, Recall@10%=0.2338, Lift@10%=2.3377, PR-AUC=0.5236, ROC-AUC=0.7236, and Brier=0.1689. The baseline captured 19 positives in the same budget. These are retrospective validation results only; no 2023 performance claim is made.

## Calibration and score interpretation

Uncalibrated, sigmoid, and isotonic methods were evaluated using 2020 base training, 2021 calibration, and 2022 validation. Neither calibrated method meaningfully improved the configured probability-quality policy while preserving ranking performance. The selected score is therefore an uncalibrated ranking output; it is not a confirmed probability of violation.

## Explainability, monitoring, and oversight

The dashboard provides deterministic candidate-level explanation and global evidence views. Monitoring compares the candidate population with historical references and exposes raw/operational drift separately. Human reviewers may override model priority and record rationale. The system must not autonomously trigger inspection or enforcement.

## Known limitations and future evaluation

The sample is California-only, not national or complete-workplace coverage. The 2023 candidate period has no outcomes available to the workflow, so future complete labels are required to evaluate frozen predictions, assess outcome fairness where data permits, and make governed retraining decisions.

## Version and lineage

Source snapshot: `edbd4bd813ed8e1dbaba9e1c`; feature version: `day2-historical-v1`; selected model: `exp_05_random_forest`; calibration method: `uncalibrated`.
