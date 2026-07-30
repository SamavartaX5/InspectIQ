# InspectIQ project metrics

This is the documentation source of truth for values extracted from committed reports. All validation metrics refer to the labelled 2022 validation population unless stated otherwise.

| Area | Value | Population / period | Source report | Interpretation and limitation |
| --- | --- | --- | --- | --- |
| Complete labels | 2,100 | California foundation | `data_foundation_report.json` | 552 positive, 1,548 negative; not all retrieved inspections |
| Unknown outcomes excluded | 900 | California foundation | `data_foundation_report.json` | Unknown/incomplete retrievals were not assumed negative |
| Positive rate | 26.29% | Complete labels | `data_foundation_report.json` | Labelled subset only |
| Split counts | 1,200 / 600 / 300 | 2020–21 train / 2022 validation / 2023 candidate | `data_foundation_report.json`, `model_comparison_report.json` | 2023 candidate has no workflow outcomes |
| Baseline top-10% | 19 positives; P=0.3167, R=0.1234, lift=1.2338 | 60 of 600 validation rows | `baseline_report.json` | Retrospective baseline comparison |
| Selected top-10% | 36 positives; P=0.6000, R=0.2338, lift=2.3377 | 60 of 600 validation rows | `model_comparison_report.json` | Random Forest validation result |
| Selected PR-AUC / ROC-AUC / Brier | 0.5236 / 0.7236 / 0.1689 | 2022 validation | `model_comparison_report.json` | Retrospective; score remains uncalibrated |
| Recall@10% improvement | +0.1104 absolute; 89.47% relative | 2022 validation | `model_comparison_report.json` | 36 versus 19 captured validation positives |
| Calibration selection | uncalibrated | 2020 → 2021 → 2022 study | `calibration_report.json` | Sigmoid and isotonic did not meaningfully qualify |
| MLflow runs | 8 Day 3; 3 Day 4; 11 reused | Local SQLite tracking | `mlflow_tracking_report.json` | Deterministic logical-run reuse |
| Candidate ranking | 300 rows; 15 / 30 / 60 at 5% / 10% / 20% | Unlabelled 2023 candidate batch | `batch_prediction_report.json` | Ranking count only, not performance |
| Monitoring | pipeline PASS; health WARNING | 2023 candidate vs historical references | `monitoring_report.json` | Two raw critical cumulative-history drifts remain visible; no performance conclusion |
| Release checks | CI PASS; local PASS | Release validation | `release_validation_report.json` | Local mode validates frozen artifacts when present |
| Docker contract | Python 3.13 slim, non-root `appuser`, port 8501 | Runtime packaging | `Dockerfile`, `docker-compose.yml` | Local artifacts are mounted read-only |

The project test suite contains 90 tests after the documentation and release-validation checks were added. This count is an engineering check, not a performance metric.
