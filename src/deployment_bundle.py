"""Build and validate the minimal, read-only InspectIQ public-demo bundle."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.batch_prediction import OUTPUT_COLUMNS, PROHIBITED_COLUMNS, TARGET
from src.explanations import training_references
from src.path_utils import RelativePathError, resolve_report_path


class DeploymentBundleError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentBundleError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DeploymentBundleError(f"Expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeploymentBundleError(message)


def _resolve(root: Path, value: str, field: str) -> Path:
    try:
        return resolve_report_path(value, root)
    except RelativePathError as exc:
        raise DeploymentBundleError(f"Invalid {field}: {exc}") from exc


def _config(root: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load((root / "config" / "deployment_config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DeploymentBundleError(f"Invalid deployment configuration: {exc}") from exc
    _require(isinstance(config, dict) and isinstance(config.get("reports"), list), "Deployment configuration is incomplete.")
    return config


def _reports(root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {Path(name).name: _read_json(root / name) for name in config["reports"]}
    required = {"batch_prediction_report.json", "model_comparison_report.json", "calibration_report.json", "mlflow_tracking_report.json", "feature_engineering_report.json", "monitoring_report.json"}
    _require(required == set(result), "Deployment report set is incomplete.")
    return result


def _validate_source(root: Path, config: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    batch = reports["batch_prediction_report.json"]
    calibration = reports["calibration_report.json"]
    feature = reports["feature_engineering_report.json"]
    monitoring = reports["monitoring_report.json"]
    comparison = reports["model_comparison_report.json"]
    for name, report in reports.items():
        _require(report.get("status") == "PASS", f"{name} must have status PASS.")
    snapshot, version = batch.get("source_snapshot_id"), batch.get("feature_version")
    _require(snapshot and all((report.get("source_snapshot_id"), report.get("feature_version")) == (snapshot, version) for report in (calibration, feature, comparison, monitoring)), "Source snapshot ID or feature version mismatch.")
    _require(batch.get("selected_day4_method") == "uncalibrated" and calibration.get("selected_calibration_method") == "uncalibrated", "Deployment must retain the uncalibrated final method.")
    _require(monitoring.get("monitoring_health") in config["allowed_monitoring_health_values"], "Monitoring health is invalid.")
    safety_checks = {
        "current_labels_accessed": monitoring.get("current_labels_accessed") is False and batch.get("labels_accessed") is False,
        "current_performance_metrics_calculated": monitoring.get("current_performance_metrics_calculated") is False and batch.get("performance_metrics_calculated") is False,
        "outcome_fairness_metrics_calculated": monitoring.get("outcome_fairness_metrics_calculated") is False,
        "prediction_artifact_modified": monitoring.get("prediction_artifact_modified") is False,
        "automatic_enforcement": monitoring.get("automatic_enforcement") is False and batch.get("automatic_enforcement") is False,
    }
    _require(all(safety_checks.values()), "Deployment source reports violate candidate safety flags.")
    flags = {
        "current_labels_accessed": False,
        "current_performance_metrics_calculated": False,
        "outcome_fairness_metrics_calculated": False,
        "prediction_artifact_modified": False,
        "automatic_enforcement": False,
    }
    ranked = _resolve(root, batch["ranked_output_path"], "ranked output path")
    top = _resolve(root, batch["top_10_output_path"], "top-10 output path")
    model = _resolve(root, batch["model_artifact_path"], "model artifact path")
    feature_train = _resolve(root, feature["output_paths"]["train"], "training feature path")
    feature_manifest = _resolve(root, feature["output_paths"]["manifest"], "feature manifest path")
    monitoring_manifest = _resolve(root, monitoring["monitoring_manifest_path"], "monitoring manifest path")
    review = _resolve(root, monitoring["review_worksheet_path"], "review worksheet path")
    future = _resolve(root, monitoring["future_outcome_template_path"], "future outcome template path")
    paths = {"ranked": ranked, "top_10": top, "model": model, "feature_train": feature_train, "feature_manifest": feature_manifest, "monitoring_manifest": monitoring_manifest, "review": review, "future": future}
    _require(all(path.is_file() for path in paths.values()), "A required frozen runtime artifact is missing.")
    _require(_sha(ranked) == batch["ranked_output_hash"] and _sha(top) == batch["top_10_output_hash"] and _sha(model) == batch["model_artifact_hash"], "Prediction or model hash does not match its report.")
    manifest = _read_json(ranked.parent / "prediction_manifest.json")
    _require(manifest.get("output_hashes", {}).get(ranked.name) == _sha(ranked) and manifest.get("output_hashes", {}).get(top.name) == _sha(top), "Prediction hashes do not match the prediction manifest.")
    monitored = _read_json(monitoring_manifest)
    _require(monitored.get("output_hashes", {}).get(review.name) == _sha(review) and monitored.get("output_hashes", {}).get(future.name) == _sha(future), "Monitoring hashes do not match its manifest.")
    feature_data = _read_json(feature_manifest)
    _require(feature_data.get("file_hashes", {}).get(feature_train.name) == _sha(feature_train), "Training reference hash does not match the feature manifest.")
    ranked_frame, top_frame = pd.read_csv(ranked), pd.read_csv(top)
    _require(list(ranked_frame.columns) == OUTPUT_COLUMNS and list(top_frame.columns) == OUTPUT_COLUMNS, "Candidate output schema is invalid.")
    _require(len(ranked_frame) == 300 and len(top_frame) == 30 and ranked_frame.activity_nr.notna().all() and ranked_frame.activity_nr.is_unique, "Candidate row or activity ID contract is invalid.")
    _require(TARGET not in ranked_frame and not (PROHIBITED_COLUMNS & set(ranked_frame.columns)), "Candidate output contains target or outcome fields.")
    scores = ranked_frame.raw_risk_score.astype(float)
    _require(scores.map(math.isfinite).all() and scores.between(0, 1).all(), "Candidate scores are not finite values in [0, 1].")
    _require(ranked_frame["rank"].tolist() == list(range(1, 301)) and top_frame.activity_nr.astype(str).tolist() == ranked_frame.head(30).activity_nr.astype(str).tolist(), "Frozen ranks are not deterministic or unchanged.")
    _require(callable(getattr(joblib.load(model), "predict_proba", None)), "Final candidate model cannot provide explanations.")
    return {"paths": paths, "flags": flags, "batch": batch, "feature": feature, "monitoring": monitoring, "prediction_manifest": manifest}


def _copy(source: Path, root: Path, temporary: Path, entries: dict[str, str]) -> None:
    relative = source.relative_to(root).as_posix()
    destination = temporary / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    entries[relative] = _sha(destination)


def _bundle_valid(bundle: Path) -> bool:
    try:
        validate_bundle(bundle)
        return True
    except DeploymentBundleError:
        return False


def build_bundle(root: Path | str = ".") -> tuple[Path, bool, dict[str, Any]]:
    root = Path(root).resolve(); config = _config(root); reports = _reports(root, config); source = _validate_source(root, config, reports)
    bundle = root / config["bundle_directory"]
    if bundle.exists() and _bundle_valid(bundle):
        return bundle, True, _read_json(bundle / "deployment_manifest.json")
    temporary = Path(tempfile.mkdtemp(prefix="deploy_bundle.tmp-", dir=root))
    try:
        entries: dict[str, str] = {}
        for name in config["reports"]:
            report_source = root / name
            if name.endswith("feature_engineering_report.json"):
                report = dict(reports["feature_engineering_report.json"])
                training = pd.read_csv(source["paths"]["feature_train"])
                references = training_references(training)
                reference_relative = "data/deployment/training_references.json"
                reference_path = temporary / reference_relative; reference_path.parent.mkdir(parents=True, exist_ok=True)
                reference_path.write_text(json.dumps(references, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                report["deployment_training_references_path"] = reference_relative
                report["deployment_training_references_hash"] = _sha(reference_path)
                target = temporary / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                entries[name] = _sha(target); entries[reference_relative] = _sha(reference_path)
            elif name.endswith("monitoring_report.json"):
                report = dict(reports["monitoring_report.json"])
                report["deployment_future_outcome_template_excluded"] = True
                report.pop("future_outcome_template_path", None); report.pop("future_outcome_template_hash", None)
                target = temporary / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                entries[name] = _sha(target)
            else:
                _copy(report_source, root, temporary, entries)
        for key in ("ranked", "top_10", "model", "monitoring_manifest", "review"):
            _copy(source["paths"][key], root, temporary, entries)
        deployment_monitoring_manifest = temporary / source["paths"]["monitoring_manifest"].relative_to(root)
        monitored = _read_json(deployment_monitoring_manifest)
        monitored.get("output_hashes", {}).pop(source["paths"]["future"].name, None)
        monitored["deployment_future_outcome_template_excluded"] = True
        deployment_monitoring_manifest.write_text(json.dumps(monitored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        entries[deployment_monitoring_manifest.relative_to(temporary).as_posix()] = _sha(deployment_monitoring_manifest)
        _copy(source["paths"]["ranked"].parent / "prediction_manifest.json", root, temporary, entries)
        manifest = {"deployment_bundle_version": config["deployment_bundle_version"], "source_release_version": config["source_release_version"], "source_snapshot_id": source["batch"]["source_snapshot_id"], "feature_version": source["batch"]["feature_version"], "prediction_run_id": source["batch"]["prediction_run_id"], "monitoring_run_id": source["monitoring"]["monitoring_run_id"], "selected_experiment": source["batch"]["selected_day3_experiment"], "calibration_method": source["batch"]["selected_day4_method"], "files": dict(sorted(entries.items())), "model_included": True, "safety_flags": source["flags"], "limitations": ["Candidate ranking is advisory and requires human review.", "Scores are uncalibrated model outputs, not verified probabilities.", "The 2023 candidate batch has no outcome labels in this workflow.", "No automatic enforcement occurs."]}
        (temporary / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_bundle(temporary)
        if bundle.exists(): shutil.rmtree(bundle)
        temporary.replace(bundle)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return bundle, False, _read_json(bundle / "deployment_manifest.json")


def validate_bundle(bundle: Path | str | None = None) -> dict[str, Any]:
    bundle = Path(bundle or os.environ.get("INSPECTIQ_RUNTIME_ROOT", ".")).resolve()
    manifest = _read_json(bundle / "deployment_manifest.json")
    files = manifest.get("files", {})
    _require(isinstance(files, dict) and files, "Deployment manifest has no files.")
    for relative, expected in files.items():
        path = _resolve(bundle, relative, "bundle file path")
        _require(path.is_file() and _sha(path) == expected, f"Deployment bundle hash mismatch: {relative}")
    _require(all(value is False for value in manifest.get("safety_flags", {}).values()), "Deployment bundle safety flags are not all false.")
    batch = _read_json(bundle / "reports" / "batch_prediction_report.json")
    ranked = _resolve(bundle, batch["ranked_output_path"], "bundled ranked output")
    top = _resolve(bundle, batch["top_10_output_path"], "bundled top output")
    prediction = _read_json(ranked.parent / "prediction_manifest.json")
    _require(_sha(ranked) == batch["ranked_output_hash"] and _sha(top) == batch["top_10_output_hash"], "Bundled prediction hashes disagree with the report.")
    _require(prediction.get("output_hashes", {}).get(ranked.name) == _sha(ranked) and prediction.get("output_hashes", {}).get(top.name) == _sha(top), "Bundled prediction hashes disagree with the manifest.")
    ranked_frame, top_frame = pd.read_csv(ranked), pd.read_csv(top)
    _require(len(ranked_frame) == 300 and len(top_frame) == 30 and ranked_frame.activity_nr.notna().all() and ranked_frame.activity_nr.is_unique, "Bundled candidate count or activity IDs are invalid.")
    _require(TARGET not in ranked_frame and not (PROHIBITED_COLUMNS & set(ranked_frame.columns)), "Bundled candidate output includes a target or outcome field.")
    _require(ranked_frame["rank"].tolist() == list(range(1, 301)) and top_frame.activity_nr.astype(str).tolist() == ranked_frame.head(30).activity_nr.astype(str).tolist(), "Bundled candidate ranks changed.")
    scores = ranked_frame.raw_risk_score.astype(float)
    _require(scores.map(math.isfinite).all() and scores.between(0, 1).all(), "Bundled candidate scores are invalid.")
    for path in bundle.rglob("*.csv"):
        headers = set(pd.read_csv(path, nrows=0).columns)
        _require(not ({TARGET, *PROHIBITED_COLUMNS} & headers), f"Deployment CSV contains a prohibited field: {path.name}")
    forbidden = ("data/raw", "mlflow.db", "mlartifacts", "mlruns", ".env")
    _require(not any(any(token in item.lower() for token in forbidden) for item in files), "Deployment bundle contains prohibited runtime state.")
    return manifest
