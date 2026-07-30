# InspectIQ architecture

## End-to-end flow

```mermaid
flowchart LR
  A[Cached OSHA inspections and violations] --> B[Day 0 feasibility]
  B --> C[Validated labelled snapshot]
  C --> D[Chronological baseline]
  C --> E[Historical feature artifacts]
  E --> F[Day 3 model comparison]
  F --> G[Day 4 calibration study]
  G --> H[Local MLflow tracking]
  G --> I[Deterministic 2023 candidate ranking]
  I --> J[Streamlit human-review dashboard]
  I --> K[Monitoring and governance artifacts]
  K --> J
  H --> L[Release validation]
  J --> L
```

The source snapshot ID identifies the immutable validated foundation; the feature version identifies the feature schema and historical-information contract. Current lineage uses snapshot `edbd4bd813ed8e1dbaba9e1c` and `day2-historical-v1`.

## Chronology and information boundaries

```mermaid
flowchart LR
  T[2020 base training] --> U[2021 training and calibration]
  U --> V[2022 labelled validation]
  V --> W[2023 locked candidate: no target]
  T -. prior history only .-> V
  U -. training plus validation history only .-> W
  W -. no label, fitting, or evaluation feedback .-> W
```

Training contains 2020–2021 labels; validation is 2022. The candidate batch is 2023 and has no target column. Strictly-prior feature construction prevents a row, future row, or same-day peer from leaking target information into its features.

## Selection and lineage

The baseline uses training-period industry rates with deterministic fallback. Eight Day 3 candidates are evaluated on 2022 validation; `exp_05_random_forest` is selected. Day 4 compares uncalibrated, sigmoid, and isotonic study artifacts. The final selected package remains uncalibrated because no calibrator qualified under the probability-quality and ranking policy.

MLflow uses a local SQLite store and deterministic logical keys: 8 Day 3 runs plus 3 Day 4 runs are reusable. Batch scoring reads the selected final candidate, generates immutable ranked and top-budget CSVs, and records hashes. It neither refits nor accesses candidate outcomes.

## Dashboard runtime contract

```mermaid
flowchart TB
  Image[Docker image: app, src, config, committed reports] --> App[Streamlit dashboard]
  Data[Host data mount: read-only] --> App
  Artifacts[Host artifacts mount: read-only] --> App
  Reports[Host reports mount: read-only] --> App
  App --> Review[Human review, rationale, override]
  App --> Evidence[Historical evidence and monitoring]
```

The dashboard consumes existing artifacts and reports. It never retrieves data, starts a training job, recalibrates, or regenerates candidates at startup. Missing required mounted artifacts produce an actionable validation error.

## Monitoring and release architecture

Monitoring compares 2023 candidate features with 2022 validation (primary reference) and 2020–2021 training (secondary reference). Data drift and score-distribution drift are descriptive; neither proves current performance drift without outcomes. Governance outputs include review and future-outcome templates, with hashes for auditability.

`run_release_validation.py --mode ci` checks source, configuration, committed reports, imports, workflow, Docker contract, documentation, and safety without local artifacts. `--mode local` additionally verifies frozen artifact paths, hashes, counts, and governance templates. The Docker image intentionally excludes raw/processed data, MLflow state, models, predictions, and monitoring artifacts; Compose mounts them read-only for local runtime.

## Safety boundaries

InspectIQ is candidate ranking for human review. It does not assert a violation, provide a calibrated probability, join 2023 labels, calculate current performance or outcome fairness, deploy an enforcement service, or automatically trigger enforcement.
