"""Offline Day 5B candidate ranking; this module never loads outcome data."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


TARGET = "serious_violation_found"
META_COLUMNS = ["activity_nr", "open_date"]
MODEL_COLUMNS = [
    "naics_group", "insp_type", "insp_scope", "owner_type", "safety_hlth",
    "nr_in_estab", "open_month", "industry_prior_inspection_count",
    "industry_prior_positive_count", "industry_prior_positive_rate_smoothed",
    "industry_history_status",
]
INPUT_COLUMNS = META_COLUMNS + MODEL_COLUMNS
OUTPUT_COLUMNS = [
    "rank", "activity_nr", "open_date", "raw_risk_score", "score_percentile",
    "review_priority", "top_5_percent_flag", "top_10_percent_flag",
    "top_20_percent_flag", *MODEL_COLUMNS,
]
PROHIBITED_COLUMNS = {
    TARGET, "citation_id", "viol_type", "delete_flag", "current_penalty",
    "initial_penalty", "issuance_date", "contest_date", "final_order_date",
    "close_case_date", "close_conf_date", "case_mod_date", "why_no_insp",
}


class PredictionError(RuntimeError):
    """The candidate-only input or generated ranking artifact is invalid."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionError(f"Invalid JSON file: {path}") from exc


def read_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PredictionError(f"Invalid prediction configuration: {path}") from exc
    if not isinstance(value, dict):
        raise PredictionError("Prediction configuration must be a mapping.")
    fractions = value.get("top_fractions")
    if not isinstance(fractions, dict) or set(fractions) != {"top_5_percent", "top_10_percent", "top_20_percent"}:
        raise PredictionError("Prediction configuration must define the three review fractions.")
    if [fractions[key] for key in ("top_5_percent", "top_10_percent", "top_20_percent")] != [0.05, 0.10, 0.20]:
        raise PredictionError("Review fractions must be 5%, 10%, and 20%.")
    if not value.get("output_root"):
        raise PredictionError("Prediction configuration is missing output_root.")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def resolve(path_value: str | Path, relative_to: Path | None = None) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() or relative_to is None else relative_to / path


def expected_model_columns(model: Any) -> list[str]:
    columns = getattr(model, "feature_names_in_", None)
    if columns is None:
        return MODEL_COLUMNS.copy()
    columns = [str(value) for value in columns]
    if set(columns) != set(MODEL_COLUMNS) or len(columns) != len(MODEL_COLUMNS):
        raise PredictionError("Final model feature contract is incompatible with the Day 2 candidate schema.")
    return columns


def positive_class_index(model: Any) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("model"), "classes_", None)
    if classes is None:
        raise PredictionError("Final candidate model does not expose class metadata.")
    classes = list(classes)
    if 1 not in classes:
        raise PredictionError("Final candidate model has no positive class labelled 1.")
    return classes.index(1)


def load_and_validate_inputs(
    calibration_path: Path, feature_report_path: Path, config_path: Path,
) -> dict[str, Any]:
    calibration = read_json(calibration_path)
    feature_report = read_json(feature_report_path)
    config = read_config(config_path)
    if calibration.get("status") != "PASS":
        raise PredictionError("Calibration report must have status PASS.")
    if feature_report.get("status") != "PASS":
        raise PredictionError("Feature engineering report must have status PASS.")
    snapshot = calibration.get("source_snapshot_id")
    version = calibration.get("feature_version")
    if not snapshot or not version or (snapshot, version) != (feature_report.get("source_snapshot_id"), feature_report.get("feature_version")):
        raise PredictionError("Calibration and feature report snapshot/version are incompatible.")
    if any(calibration.get(key) for key in ("locked_test_labels_accessed", "locked_test_metrics_calculated", "locked_test_predictions_created")):
        raise PredictionError("Calibration report violates locked-candidate safeguards.")
    model_path = resolve(calibration.get("final_candidate_artifact_path", ""))
    if not model_path.is_file():
        raise PredictionError(f"Final candidate artifact is missing: {model_path}")
    input_path = resolve(feature_report.get("output_paths", {}).get("test_locked", ""))
    manifest_path = resolve(feature_report.get("output_paths", {}).get("manifest", ""))
    if not input_path.is_file() or not manifest_path.is_file():
        raise PredictionError("Locked candidate feature file or manifest is missing.")
    manifest = read_json(manifest_path)
    if (manifest.get("source_snapshot_id"), manifest.get("feature_version")) != (snapshot, version):
        raise PredictionError("Locked candidate manifest snapshot/version is incompatible.")
    input_hash = sha256(input_path)
    if manifest.get("file_hashes", {}).get(input_path.name) != input_hash:
        raise PredictionError("Locked candidate feature hash does not match its manifest.")
    if feature_report.get("output_hashes", {}).get(input_path.name) != input_hash:
        raise PredictionError("Locked candidate feature hash does not match its report.")
    candidates = pd.read_csv(input_path)
    # Day 2 manifests identify splits by logical name; accept a filename key too
    # for forwards-compatible manifests, while requiring one authoritative schema.
    expected_columns = manifest.get("output_columns", {}).get("test_locked", manifest.get("output_columns", {}).get(input_path.name))
    if expected_columns is not None and list(expected_columns) != INPUT_COLUMNS:
        raise PredictionError("Locked candidate manifest schema differs from the expected candidate schema.")
    if list(candidates.columns) != INPUT_COLUMNS:
        raise PredictionError("Locked candidate schema differs from the expected candidate schema.")
    expected_rows = manifest.get("output_row_counts", {}).get("test_locked", manifest.get("output_row_counts", {}).get(input_path.name))
    report_rows = feature_report.get("splits", {}).get("test_locked", {}).get("output_row_count")
    if len(candidates) == 0 or expected_rows != len(candidates) or report_rows != len(candidates):
        raise PredictionError("Locked candidate row count does not match its manifest and feature report.")
    if TARGET in candidates or PROHIBITED_COLUMNS.intersection(candidates.columns):
        raise PredictionError("Locked candidate input contains a prohibited target or outcome column.")
    if candidates["activity_nr"].isna().any() or not candidates["activity_nr"].is_unique:
        raise PredictionError("activity_nr must be present, non-null, and unique.")
    dates = pd.to_datetime(candidates["open_date"], errors="coerce")
    if dates.isna().any():
        raise PredictionError("open_date must be present and valid for every candidate.")
    return {
        "calibration": calibration, "feature_report": feature_report, "config": config,
        "snapshot": snapshot, "version": version, "model_path": model_path,
        "model_hash": sha256(model_path), "input_path": input_path, "input_hash": input_hash,
        "manifest_path": manifest_path, "candidates": candidates, "dates": dates,
    }


def prediction_run_id(context: dict[str, Any], config_path: Path) -> str:
    identity = {
        "model_hash": context["model_hash"], "input_hash": context["input_hash"],
        "config_hash": sha256(config_path), "source_snapshot_id": context["snapshot"],
        "feature_version": context["version"],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def score_and_rank(candidates: pd.DataFrame, model: Any, fractions: dict[str, float]) -> tuple[pd.DataFrame, dict[str, int], dict[str, float]]:
    if not callable(getattr(model, "predict_proba", None)):
        raise PredictionError("Final candidate artifact must implement predict_proba.")
    model_columns = expected_model_columns(model)
    positive_index = positive_class_index(model)
    probabilities = np.asarray(model.predict_proba(candidates.loc[:, model_columns]))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(candidates) or positive_index >= probabilities.shape[1]:
        raise PredictionError("Final candidate model returned an invalid probability matrix.")
    scores = np.asarray(probabilities[:, positive_index], dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise PredictionError("Candidate raw risk scores must be finite values in [0, 1].")
    result = candidates.copy()
    result["raw_risk_score"] = scores
    result["_activity_sort"] = result["activity_nr"].astype(str)
    result = result.sort_values(["raw_risk_score", "_activity_sort"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["score_percentile"] = ((len(result) - result["rank"] + 1) * 100 / len(result)).round(6)
    counts = {name: max(1, math.ceil(len(result) * fraction)) for name, fraction in fractions.items()}
    result["top_5_percent_flag"] = result["rank"] <= counts["top_5_percent"]
    result["top_10_percent_flag"] = result["rank"] <= counts["top_10_percent"]
    result["top_20_percent_flag"] = result["rank"] <= counts["top_20_percent"]
    result["review_priority"] = np.select(
        [result["top_5_percent_flag"], result["top_10_percent_flag"], result["top_20_percent_flag"]],
        ["highest_priority", "high_priority", "elevated_priority"], default="standard_priority",
    )
    result = result.loc[:, OUTPUT_COLUMNS]
    summary = {
        "minimum": float(scores.min()), "maximum": float(scores.max()), "mean": float(scores.mean()),
        "median": float(np.median(scores)), "standard_deviation": float(scores.std(ddof=0)),
    }
    return result, counts, summary


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def valid_artifact(directory: Path, context: dict[str, Any], run_id: str, config_path: Path) -> dict[str, Any] | None:
    names = ("ranked_candidates.csv", "top_10_percent_candidates.csv", "prediction_manifest.json")
    if not directory.is_dir() or not all((directory / name).is_file() for name in names):
        return None
    try:
        manifest = read_json(directory / "prediction_manifest.json")
        expected = {"prediction_run_id": run_id, "source_snapshot_id": context["snapshot"], "feature_version": context["version"], "model_artifact_hash": context["model_hash"], "input_feature_hash": context["input_hash"], "prediction_config_hash": sha256(config_path)}
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        ranked, top = directory / "ranked_candidates.csv", directory / "top_10_percent_candidates.csv"
        hashes = manifest.get("output_hashes", {})
        if hashes.get(ranked.name) != sha256(ranked) or hashes.get(top.name) != sha256(top):
            return None
        ranked_frame, top_frame = pd.read_csv(ranked), pd.read_csv(top)
        if list(ranked_frame.columns) != OUTPUT_COLUMNS or list(top_frame.columns) != OUTPUT_COLUMNS:
            return None
        if len(ranked_frame) != len(context["candidates"]) or len(top_frame) != manifest.get("top_counts", {}).get("top_10_percent"):
            return None
        if TARGET in ranked_frame or PROHIBITED_COLUMNS.intersection(ranked_frame.columns):
            return None
        return manifest
    except (OSError, ValueError, PredictionError):
        return None


def build_manifest(context: dict[str, Any], run_id: str, config_path: Path, ranked: pd.DataFrame, counts: dict[str, int], summary: dict[str, float], runtime: float) -> dict[str, Any]:
    return {
        "prediction_run_id": run_id, "source_snapshot_id": context["snapshot"], "feature_version": context["version"],
        "model_artifact_path": str(context["model_path"]), "model_artifact_hash": context["model_hash"],
        "input_feature_path": str(context["input_path"]), "input_feature_hash": context["input_hash"],
        "prediction_config_hash": sha256(config_path), "row_count": len(ranked), "score_summary": summary,
        "top_counts": counts, "selected_calibration_method": context["calibration"]["selected_calibration_method"],
        "final_calibration_applied": context["calibration"]["final_calibration_applied"],
        "score_interpretation": "raw_risk_score is an uncalibrated model output used only to rank this supplied candidate batch.",
        "ranking_rule": "raw_risk_score descending, then activity_nr ascending; review budgets use ceil-based 5%, 10%, and 20% cutoffs.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": round(runtime, 4),
        "locked_labels_accessed": False, "locked_metrics_calculated": False, "automatic_enforcement_initiated": False,
    }


def report_from_manifest(context: dict[str, Any], manifest: dict[str, Any], directory: Path, reused: bool, runtime: float) -> dict[str, Any]:
    ranked, top = directory / "ranked_candidates.csv", directory / "top_10_percent_candidates.csv"
    priorities = pd.read_csv(ranked)["review_priority"].value_counts().to_dict()
    return {
        "status": "PASS", "source_snapshot_id": context["snapshot"], "feature_version": context["version"],
        "prediction_run_id": manifest["prediction_run_id"], "model_artifact_path": manifest["model_artifact_path"],
        "model_artifact_hash": manifest["model_artifact_hash"], "selected_day3_experiment": context["calibration"]["selected_day3_experiment"],
        "selected_day4_method": manifest["selected_calibration_method"], "final_calibration_applied": manifest["final_calibration_applied"],
        "locked_candidate_input_path": manifest["input_feature_path"], "locked_candidate_input_hash": manifest["input_feature_hash"],
        "input_row_count": len(context["candidates"]), "input_column_count": len(context["candidates"].columns),
        "open_date_range": {"minimum": str(context["dates"].min().date()), "maximum": str(context["dates"].max().date())},
        "score_summary": manifest["score_summary"], "score_range_checks": {"finite": True, "within_zero_to_one": True},
        "ranked_output_path": str(ranked), "ranked_output_hash": sha256(ranked), "top_10_output_path": str(top), "top_10_output_hash": sha256(top),
        "top_counts": manifest["top_counts"], "review_priority_counts": priorities, "artifacts_reused": reused,
        "labels_accessed": False, "performance_metrics_calculated": False, "evaluation_performed": False,
        "automatic_enforcement": False, "runtime_seconds": round(runtime, 4),
        "limitations": [
            "No labels were accessed for the 2023 candidate batch.", "No test performance metrics were calculated.",
            "Scores are uncalibrated model outputs.", "A score is not a verified probability of a serious violation.",
            "Historical OSHA inspection data is selection-biased.", "The model ranks only the supplied candidate batch.",
            "Rankings support human review and do not automatically initiate enforcement.",
            "A later labelled evaluation is required to measure real out-of-time performance.",
        ],
    }


def run(calibration_path: Path = Path("reports/calibration_report.json"), feature_report_path: Path = Path("reports/feature_engineering_report.json"), config_path: Path = Path("config/prediction_config.yaml")) -> dict[str, Any]:
    started = time.perf_counter()
    context = load_and_validate_inputs(calibration_path, feature_report_path, config_path)
    model = joblib.load(context["model_path"])
    if not callable(getattr(model, "predict_proba", None)):
        raise PredictionError("Final candidate artifact must implement predict_proba.")
    run_id = prediction_run_id(context, config_path)
    directory = resolve(context["config"]["output_root"]) / run_id
    existing = valid_artifact(directory, context, run_id, config_path)
    if existing is not None:
        return report_from_manifest(context, existing, directory, True, time.perf_counter() - started)
    if directory.exists():
        directory.rename(directory.with_name(directory.name + ".invalid-" + uuid.uuid4().hex[:8]))
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.mkdir()
        ranked, counts, summary = score_and_rank(context["candidates"], model, context["config"]["top_fractions"])
        ranked_path, top_path = temporary / "ranked_candidates.csv", temporary / "top_10_percent_candidates.csv"
        write_csv(ranked, ranked_path)
        write_csv(ranked.head(counts["top_10_percent"]), top_path)
        manifest = build_manifest(context, run_id, config_path, ranked, counts, summary, time.perf_counter() - started)
        manifest["output_hashes"] = {ranked_path.name: sha256(ranked_path), top_path.name: sha256(top_path)}
        atomic_json(temporary / "prediction_manifest.json", manifest)
        if valid_artifact(temporary, context, run_id, config_path) is None:
            raise PredictionError("Generated prediction artifact failed its integrity validation.")
        temporary.replace(directory)
        return report_from_manifest(context, manifest, directory, False, time.perf_counter() - started)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
