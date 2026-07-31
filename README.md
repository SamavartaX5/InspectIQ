# InspectIQ

InspectIQ is a reproducible, advisory decision-support project that ranks a supplied OSHA inspection candidate batch for human review using historically validated, leakage-safe features.

## Problem and decision-support framing

Review capacity is limited. InspectIQ orders only the supplied candidates so reviewers can focus a fixed review budget; it does **not** create a complete workplace-risk registry, confirm violations, or initiate enforcement. Every recommendation remains subject to human review and documented override.

## Key verified results

All performance figures below are retrospective results on the labelled **2022 validation period** (600 rows, 154 positives), not claims about the unlabelled 2023 candidate batch.

| Measure | 2022 baseline | Selected Random Forest | Interpretation |
| --- | ---: | ---: | --- |
| Positives captured at top 10% | 19 / 60 | 36 / 60 | 17 additional validation positives at the same review budget |
| Precision@10% | 0.3167 | 0.6000 | Share of reviewed 2022 validation rows labelled positive |
| Recall@10% | 0.1234 | 0.2338 | Share of 2022 validation positives captured |
| Lift@10% | 1.2338× | 2.3377× | Ranking concentration relative to the 2022 base rate |
| PR-AUC | 0.3076 | 0.5236 | Validation ranking quality |
| ROC-AUC | 0.5873 | 0.7236 | Validation discrimination summary |

The 2023 batch contains 300 supplied candidates with no outcomes loaded by this workflow. Its uncalibrated model outputs are advisory ranking scores, **not calibrated probabilities** and not a current-performance claim.

## Dashboard overview

The Streamlit dashboard provides Review Queue, Candidate Detail, Model Evidence, Monitoring & Governance, and Data & Limitations pages. It supports review-budget filtering, deterministic candidate explanations, evidence display, monitoring summaries, and governance-template downloads. It never silently regenerates artifacts or contacts external APIs on startup.

## System architecture

The pipeline uses cached Day 0 acquisition, a validated labelled foundation, chronological splits, historical features, model comparison, calibration study, local MLflow tracking, deterministic batch ranking, review templates, monitoring, and release checks. See [architecture](docs/architecture.md) for diagrams and lineage.

## Data and label construction

The source is the U.S. Department of Labor OSHA inspection and violation endpoints. California inspections use `activity_nr` as the join key and `open_date` as the candidate date. A positive label means at least one non-deleted Serious, Willful, or Repeat violation. Deleted rows are excluded; incomplete or unknown violation retrieval remains excluded rather than becoming a negative label.

The immutable foundation snapshot is `edbd4bd813ed8e1dbaba9e1c`. It contains 2,100 complete labels: 552 positive and 1,548 negative (26.29% positive); 900 unknown outcomes are excluded. See [data card](docs/data_card.md).

## Chronological evaluation design

Training uses 2020–2021 (1,200 labelled rows), validation uses 2022 (600 labelled rows), and the locked 2023 candidate period contains 300 rows without a target. Information flows forward only. The candidate period is not used for training, calibration, or reported performance.

## Leakage-safe feature engineering

Feature version `day2-historical-v1` uses strictly prior history: a row cannot contribute to its own feature, same-day rows cannot affect one another, validation uses training history only, and candidates use training-plus-validation history only. Industry history is retained; establishment history is omitted because no defensible establishment identifier is available.

## Baseline versus selected model

The baseline uses training-period industry rates with deterministic fallbacks. Eight candidate models were compared on the 2022 validation set; `exp_05_random_forest` was selected by Recall@10%, then precision, lift, PR-AUC, Brier score, and model simplicity. Its Random Forest uses 200 trees, depth 8, minimum leaf size 5, and random state 42.

## Calibration decision

Uncalibrated, sigmoid, and isotonic methods were studied using 2020 base training, 2021 calibration, and 2022 validation. Neither calibrated method meaningfully improved probability quality while preserving ranking utility, so the final package is uncalibrated. Scores must be interpreted as model ranking outputs, not confirmed probabilities.

## MLflow experiment tracking

Local SQLite MLflow records eight Day 3 model experiments and three Day 4 calibration-study runs. A second execution reused all 11 deterministic logical runs. No remote tracker, secret, or DagsHub service is required.

## Batch scoring and human review

The frozen 2023 candidate batch has 300 ranked rows: top 5% = 15, top 10% = 30, and top 20% = 60. The dashboard and governance worksheets support human rationale, override, and escalation fields. No automatic enforcement occurs.

## Monitoring and governance

Monitoring compares the 2023 candidate feature population to 2022 validation (primary) and 2020–2021 training (secondary). The monitoring pipeline is PASS and operational health is WARNING: two cumulative-history features retain raw critical drift but are operational warnings due to expected temporal accumulation. Score drift is not performance drift; complete future outcome labels are needed for performance and outcome-fairness evaluation. Details: [governance](docs/governance.md).

## Repository structure

```text
app/                 Streamlit review dashboard
config/              Versioned pipeline and release configuration
docs/                Architecture, cards, governance, and portfolio material
reports/             Committed validation summaries and schemas
src/                 Pipeline, monitoring, governance, and release checks
tests/               Synthetic and offline unit tests
run_*.py             Explicit pipeline commands
```

## Local setup

PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Platform-neutral:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the pipeline

Run commands only when intentionally reproducing the workflow; they can change generated, ignored artifacts. The exact order and network/offline behaviour are documented in [reproducibility](docs/reproducibility.md).

## Running tests and release validation

```powershell
python -m unittest discover -s tests -t . -v
python -m compileall app src scripts tests run_release_validation.py
python run_release_validation.py --mode ci
python run_release_validation.py --mode local
```

CI mode is clean-checkout safe and does not require ignored artifacts. Local mode verifies the frozen local model, predictions, and monitoring artifacts without regenerating them.

## Running Streamlit and Docker

```powershell
python -m streamlit run app/streamlit_app.py
docker build --tag inspectiq:release .
docker compose up
```

The Docker image contains code, configuration, and committed reports only. `docker-compose.yml` mounts locally generated `data`, `artifacts`, and `reports` read-only. Those inputs must exist locally; the container will not download or regenerate them.

### Preparing the public-demo image

The public-demo package is prepared locally from frozen runtime artifacts; it does not publish an image or provide a public URL.

```powershell
python scripts/build_deployment_bundle.py
python run_deployment_validation.py
docker build -f Dockerfile.deploy -t inspectiq-demo:v1.0.1 .
```

`deploy_bundle/` is ignored. It contains only the candidate queue, its manifests, the final model needed for explanations, aggregate target-free training references, monitoring review data, and dashboard reports. It excludes labels, outcomes, raw caches, MLflow state, reviewer data, and secrets.

## CI behaviour and ignored artifacts

GitHub Actions installs dependencies, checks whitespace, compiles code, runs tests, executes CI-safe release validation, and builds the image without starting the dashboard. Raw data, processed data, model artifacts, predictions, MLflow state, and monitoring artifacts are intentionally ignored: they may be private, large, and reproducible from controlled local inputs. They are not silently committed.

## Limitations and responsible use

Historical OSHA inspections are selection-biased and the California sample is not a census of workplaces or OSHA activity. The early-year acquisition originally required year-balanced sampling; the project does not claim complete national coverage. There are no protected demographic attributes for outcome-fairness evaluation. InspectIQ is advisory-only, requires human review, and must not autonomously trigger inspection or enforcement.

## Future work

With complete future outcomes, evaluate frozen predictions out of time, assess outcome fairness where lawful and appropriate attributes exist, investigate data coverage, and make retraining decisions under documented governance—not from score drift alone.
