"""Validated, candidate-only data access and filtering for the Day 5C dashboard."""
from __future__ import annotations

import hashlib
import json
import math
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.batch_prediction import OUTPUT_COLUMNS, PROHIBITED_COLUMNS, TARGET
from src.explanations import global_feature_importance, local_perturbation_explanation, training_references
from src.path_utils import RelativePathError, resolve_report_path


class DashboardError(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardError(f"Invalid dashboard JSON artifact: {path}") from exc


def resolve(value: str | Path) -> Path:
    # Dashboard tests and explicit local operators may provide a trusted
    # temporary absolute config/artifact path. Report-derived paths remain
    # relative in production bundles and are resolved strictly below runtime.
    direct = Path(value)
    if direct.is_absolute():
        return direct
    try:
        return resolve_report_path(value, Path(os.environ.get("INSPECTIQ_RUNTIME_ROOT", ".")))
    except RelativePathError as exc:
        raise DashboardError(str(exc)) from exc


def read_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DashboardError(f"Invalid dashboard configuration: {path}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("reports"), dict):
        raise DashboardError("Dashboard configuration must include report paths.")
    return config


@dataclass
class DashboardContext:
    config: dict[str, Any]
    batch: dict[str, Any]
    comparison: dict[str, Any]
    calibration: dict[str, Any]
    mlflow: dict[str, Any]
    monitoring: dict[str, Any]
    prediction_manifest: dict[str, Any]
    feature_report: dict[str, Any]
    feature_manifest: dict[str, Any]
    ranked: pd.DataFrame
    top_10: pd.DataFrame
    training: pd.DataFrame | None
    training_references: dict[str, Any]
    model: Any


def _require_pass(name: str, report: dict[str, Any]) -> None:
    if report.get("status") != "PASS":
        raise DashboardError(f"{name} report must have status PASS.")


def _priority_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {key: int(value) for key, value in frame["review_priority"].value_counts().to_dict().items()}


def load_dashboard_context(config_path: Path = Path("config/dashboard_config.yaml")) -> DashboardContext:
    config = read_config(config_path)
    reports = config["reports"]
    try:
        batch = read_json(resolve(reports["batch_prediction"]))
        comparison = read_json(resolve(reports["model_comparison"]))
        calibration = read_json(resolve(reports["calibration"]))
        mlflow = read_json(resolve(reports["mlflow_tracking"]))
        monitoring_path = reports.get("monitoring")
        monitoring = read_json(resolve(monitoring_path)) if monitoring_path else {}
    except KeyError as exc:
        raise DashboardError("Dashboard configuration lacks a required report path.") from exc
    for name, report in (("Batch prediction", batch), ("Model comparison", comparison), ("Calibration", calibration), ("MLflow tracking", mlflow)):
        _require_pass(name, report)
    if monitoring:
        _require_pass("Monitoring", monitoring)
    snapshot, version = batch.get("source_snapshot_id"), batch.get("feature_version")
    if not snapshot or any((report.get("source_snapshot_id"), report.get("feature_version")) != (snapshot, version) for report in (comparison, calibration)):
        raise DashboardError("Dashboard reports have incompatible snapshot or feature versions.")
    if any(batch.get(key) for key in ("labels_accessed", "performance_metrics_calculated", "evaluation_performed", "automatic_enforcement")):
        raise DashboardError("Batch prediction report violates candidate-only dashboard safeguards.")
    if any(comparison.get(key) for key in ("locked_test_labels_accessed", "locked_test_metrics_calculated")):
        raise DashboardError("Model comparison report violates locked-candidate safeguards.")
    if any(mlflow.get(key) for key in ("locked_test_labels_accessed", "locked_test_metrics_calculated", "locked_test_predictions_created")):
        raise DashboardError("MLflow tracking report violates locked-candidate safeguards.")
    if any(calibration.get(key) for key in ("locked_test_labels_accessed", "locked_test_metrics_calculated", "locked_test_predictions_created")):
        raise DashboardError("Calibration report violates locked-candidate safeguards.")
    if monitoring and any(monitoring.get(key) for key in ("current_labels_accessed", "current_performance_metrics_calculated", "outcome_fairness_metrics_calculated", "automatic_enforcement")):
        raise DashboardError("Monitoring report violates candidate-only dashboard safeguards.")
    if monitoring and (monitoring.get("source_snapshot_id"), monitoring.get("feature_version")) != (snapshot, version):
        raise DashboardError("Monitoring report is incompatible with the prediction batch.")
    ranked_path, top_path = resolve(batch.get("ranked_output_path", "")), resolve(batch.get("top_10_output_path", ""))
    manifest_path = ranked_path.parent / "prediction_manifest.json"
    if not ranked_path.is_file() or not top_path.is_file() or not manifest_path.is_file():
        raise DashboardError("Required prediction artifacts are missing.")
    prediction_manifest = read_json(manifest_path)
    required_manifest = {"prediction_run_id": batch.get("prediction_run_id"), "source_snapshot_id": snapshot, "feature_version": version, "model_artifact_hash": batch.get("model_artifact_hash"), "input_feature_hash": batch.get("locked_candidate_input_hash")}
    if any(prediction_manifest.get(key) != value for key, value in required_manifest.items()):
        raise DashboardError("Prediction manifest disagrees with the batch prediction report.")
    if prediction_manifest.get("output_hashes", {}).get(ranked_path.name) != digest(ranked_path) or prediction_manifest.get("output_hashes", {}).get(top_path.name) != digest(top_path):
        raise DashboardError("Prediction output hash does not match its manifest.")
    ranked, top_10 = pd.read_csv(ranked_path), pd.read_csv(top_path)
    expected_rows, expected_top = config.get("expected_ranked_rows"), config.get("expected_top_10_rows")
    if list(ranked.columns) != OUTPUT_COLUMNS or list(top_10.columns) != OUTPUT_COLUMNS:
        raise DashboardError("Ranked prediction schema is invalid.")
    if len(ranked) != expected_rows or len(top_10) != expected_top or len(top_10) != batch.get("top_counts", {}).get("top_10_percent"):
        raise DashboardError("Ranked or top-10 candidate row count is invalid.")
    if ranked["activity_nr"].isna().any() or not ranked["activity_nr"].is_unique:
        raise DashboardError("Ranked candidates must have unique non-null activity IDs.")
    if TARGET in ranked or PROHIBITED_COLUMNS.intersection(ranked.columns):
        raise DashboardError("Ranked candidates contain a prohibited target or outcome field.")
    scores = ranked["raw_risk_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise DashboardError("Ranked candidate score range is invalid.")
    if ranked["rank"].tolist() != list(range(1, len(ranked) + 1)):
        raise DashboardError("Candidate ranks must be unique and sequential.")
    expected_order = ranked.assign(_activity_sort=ranked["activity_nr"].astype(str)).sort_values(["raw_risk_score", "_activity_sort"], ascending=[False, True], kind="mergesort")["activity_nr"].astype(str).tolist()
    if ranked["activity_nr"].astype(str).tolist() != expected_order:
        raise DashboardError("Ranked candidates are not in deterministic score/activity order.")
    if _priority_counts(ranked) != batch.get("review_priority_counts"):
        raise DashboardError("Candidate review-priority counts disagree with the report.")
    if top_10["activity_nr"].astype(str).tolist() != ranked.head(len(top_10))["activity_nr"].astype(str).tolist():
        raise DashboardError("Top-10 candidate queue is not the ranked queue prefix.")
    feature_report_path = resolve(reports.get("feature_engineering", "reports/feature_engineering_report.json"))
    feature_report = read_json(feature_report_path)
    _require_pass("Feature engineering", feature_report)
    if (feature_report.get("source_snapshot_id"), feature_report.get("feature_version")) != (snapshot, version):
        raise DashboardError("Feature report is incompatible with the prediction batch.")
    if feature_report.get("splits", {}).get("test_locked", {}).get("target_present") is not False:
        raise DashboardError("Feature report does not confirm the locked candidate target is absent.")
    output_paths = feature_report.get("output_paths", {})
    reference_path = feature_report.get("deployment_training_references_path")
    if reference_path:
        reference_file = resolve(reference_path)
        if not reference_file.is_file() or digest(reference_file) != feature_report.get("deployment_training_references_hash"):
            raise DashboardError("Deployment training-reference artifact is missing or invalid.")
        references = read_json(reference_file)
        training = None
    else:
        training_path, feature_manifest_path = resolve(output_paths.get("train", "")), resolve(output_paths.get("manifest", ""))
        if not training_path.is_file() or not feature_manifest_path.is_file():
            raise DashboardError("Training feature artifact or manifest is missing.")
        feature_manifest = read_json(feature_manifest_path)
        if feature_manifest.get("file_hashes", {}).get(training_path.name) != digest(training_path):
            raise DashboardError("Training feature artifact hash does not match its manifest.")
        training = pd.read_csv(training_path)
        references = training_references(training)
    model_path = resolve(batch.get("model_artifact_path", ""))
    if not model_path.is_file() or digest(model_path) != batch.get("model_artifact_hash") or model_path != resolve(calibration.get("final_candidate_artifact_path", "")):
        raise DashboardError("Final candidate model path or hash is inconsistent.")
    model = joblib.load(model_path)
    if not callable(getattr(model, "predict_proba", None)):
        raise DashboardError("Final candidate model cannot perform predict_proba.")
    return DashboardContext(config, batch, comparison, calibration, mlflow, monitoring, prediction_manifest, feature_report, feature_manifest if not reference_path else {}, ranked, top_10, training, references, model)


def filter_candidates(frame: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    filters = filters or {}
    result = frame.copy()
    mapping = {"review_priority": "review_priority", "naics_group": "naics_group", "insp_type": "insp_type", "insp_scope": "insp_scope", "owner_type": "owner_type", "safety_hlth": "safety_hlth"}
    for key, column in mapping.items():
        selected = filters.get(key)
        if selected:
            values = {str(value) for value in (selected if isinstance(selected, (list, tuple, set)) else [selected])}
            result = result[result[column].astype(str).isin(values)]
    score_range = filters.get("raw_risk_score")
    if score_range is not None:
        low, high = score_range
        result = result[result["raw_risk_score"].between(float(low), float(high), inclusive="both")]
    return result.reset_index(drop=True)


def queue_csv(frame: pd.DataFrame) -> bytes:
    if TARGET in frame or PROHIBITED_COLUMNS.intersection(frame.columns):
        raise DashboardError("Downloads may not include target or outcome fields.")
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def validation_report(context: DashboardContext, runtime: float) -> dict[str, Any]:
    explanation = local_perturbation_explanation(context.model, context.ranked.iloc[0], context.training_references)
    importance = global_feature_importance(context.model)
    queue_csv(context.ranked); queue_csv(context.top_10); queue_csv(filter_candidates(context.ranked, {"review_priority": "highest_priority"}))
    return {
        "status": "PASS", "source_snapshot_id": context.batch["source_snapshot_id"], "feature_version": context.batch["feature_version"],
        "prediction_run_id": context.batch["prediction_run_id"], "ranked_row_count": len(context.ranked), "top_10_row_count": len(context.top_10),
        "required_column_checks": True, "score_range_checks": True, "target_absent": True, "prohibited_fields_absent": True,
        "rank_order_check": True, "priority_count_check": True, "model_loaded": True, "model_refit_attempted": False,
        "global_importance_feature_count": len(importance), "global_importance_sum": float(importance["importance"].sum()),
        "local_explanation_feature_count": len(explanation), "download_generation_checks": True,
        "locked_labels_accessed": False, "performance_metrics_calculated": False, "automatic_enforcement": False,
        "runtime_seconds": round(runtime, 4),
        "limitations": [
            "No locked candidate labels were accessed.", "No out-of-time performance metrics were calculated.",
            "Scores are uncalibrated model outputs, not verified probabilities.",
            "Global and local explanations describe fitted-model behaviour, not causality.",
            "Rankings support human review and do not automatically initiate enforcement.",
        ],
    }


def validate(config_path: Path = Path("config/dashboard_config.yaml")) -> dict[str, Any]:
    started = time.perf_counter()
    return validation_report(load_dashboard_context(config_path), time.perf_counter() - started)
