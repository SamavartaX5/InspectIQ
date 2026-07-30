# Reproducibility guide

Run commands from the repository root. Generated data, artifacts, predictions, MLflow state, and monitoring outputs are ignored; do not commit them. Commands below are listed in pipeline order. Preserve a frozen release by running only tests and release validation.

## Environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Platform-neutral: `python -m venv .venv`, activate it, then `python -m pip install -r requirements.txt`.

## Pipeline commands

| Stage | Command | Network / artifact behaviour |
| --- | --- | --- |
| Day 0 feasibility | `python run_feasibility.py --offline` | Offline; cache-only, no API call. Without `--offline`, compatible caches are reused and only missing retrieval may call DOL. |
| Data foundation | `python run_data_foundation.py` | Offline transformation of cached/foundation inputs; writes ignored snapshots. |
| Baseline | `python run_baseline.py` | Offline; writes baseline artifacts. |
| Features | `python run_feature_engineering.py` | Offline; validates/reuses or safely rebuilds feature artifacts. |
| Training | `python run_training.py` | Offline; trains from local feature artifacts and writes ignored model artifacts. |
| Calibration | `python run_calibration.py` | Offline; evaluates study artifacts and packages the selected candidate. |
| MLflow | `python run_mlflow_tracking.py` | Offline local SQLite tracking; no remote MLflow/DagsHub. |
| Batch ranking | `python run_batch_prediction.py` | Offline; reads frozen candidate/model artifacts and writes deterministic rankings. |
| Monitoring | `python run_monitoring.py` | Offline; reads frozen rankings and writes monitoring/governance artifacts. |
| Dashboard validation | `python run_dashboard_validation.py` | Offline; validates dashboard inputs without serving it. |

These commands require their preceding ignored local artifacts. They are not CI prerequisites and should not be run merely to validate a frozen release.

## Tests, release checks, and dashboard

```powershell
python -m unittest discover -s tests -t . -v
python -m compileall app src scripts tests run_release_validation.py
python run_release_validation.py --mode ci
python run_release_validation.py --mode local
python -m streamlit run app/streamlit_app.py
```

`--mode ci` is clean-checkout safe: it checks source/configuration/reports/contracts but does not need ignored artifacts. `--mode local` is read-only and verifies frozen model, prediction, monitoring, and governance artifacts when present. Neither mode trains, calibrates, scores, loads current labels, or calls external APIs.

## Docker

```powershell
docker build --tag inspectiq:release .
docker compose up
```

The image intentionally excludes raw and processed data, model/prediction artifacts, MLflow state, and caches. Compose mounts `./data`, `./artifacts`, and `./reports` read-only. It must be run only where those local frozen inputs are available; startup does not download, regenerate, or contact an API.

## CI-safe commands

GitHub Actions runs compilation, tests, `python run_release_validation.py --mode ci`, and a Docker build. It uses synthetic/temporary test fixtures and does not require a DOL API key, candidate artifacts, or locked outcomes.
