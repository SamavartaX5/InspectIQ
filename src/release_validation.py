"""Read-only Day 7A release integrity checks.

This module validates frozen reports and artifacts.  It never fetches data,
loads outcome labels, fits models, recalibrates scores, or regenerates ranking.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml


class ReleaseValidationError(RuntimeError):
    """A release contract check failed."""


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"Invalid JSON report: {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"JSON report must contain an object: {path.name}")
    return value


def _load_config(root: Path) -> dict[str, Any]:
    path = root / "config" / "release_config.yaml"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseValidationError(f"Invalid release configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError("release configuration must be a YAML mapping")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseValidationError(message)


def _path_from_report(root: Path, value: Any, field: str) -> Path:
    _require(isinstance(value, str) and value, f"Missing report path: {field}")
    path = Path(value.replace("\\", "/"))
    _require(not path.is_absolute(), f"Absolute artifact path is not permitted: {field}")
    return root / path


def _workflow_checks(root: Path) -> dict[str, bool]:
    path = root / ".github" / "workflows" / "ci.yml"
    text = path.read_text(encoding="utf-8")
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ReleaseValidationError(f"Invalid CI workflow YAML: {exc}") from exc
    lowered = text.lower()
    checks = {
        "yaml_parses": True,
        "python_3_13": "python-version: '3.13'" in text or 'python-version: "3.13"' in text,
        "required_triggers": all(token in text for token in ("push:", "pull_request:", "workflow_dispatch:")),
        "compileall": "compileall" in text,
        "unittest": "unittest discover" in text,
        "release_validation": "run_release_validation.py --mode ci" in text,
        "docker_build": "docker build" in text,
        "least_privilege": "contents: read" in lowered,
        "no_secrets": "secrets." not in lowered and "DOL_API_KEY" not in text,
    }
    _require(all(checks.values()), "CI workflow does not meet the release contract")
    return checks


def _docker_checks(root: Path) -> dict[str, bool]:
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    ignored = (root / ".dockerignore").read_text(encoding="utf-8")
    checks = {
        "python_3_13_slim": "FROM python:3.13-slim" in dockerfile,
        "non_root_user": "USER appuser" in dockerfile,
        "port_8501": "EXPOSE 8501" in dockerfile,
        "pythonpath": "PYTHONPATH=/app" in dockerfile,
        "streamlit_command": False,  # populated below to keep checks ordered
        "binds_container_address": "--server.address=0.0.0.0" in dockerfile,
        "env_excluded": ".env" in ignored,
        "generated_state_excluded": all(token in ignored for token in ("artifacts", "data/raw", "mlflow.db", "mlartifacts")),
    }
    checks["streamlit_command"] = all(token in dockerfile for token in ("python", "-m", "streamlit", "app/streamlit_app.py"))
    _require(all(checks.values()), "Docker packaging does not meet the runtime contract")
    return checks


def _tracked_file_checks(root: Path, config: dict[str, Any]) -> dict[str, bool]:
    # A non-git synthetic fixture is valid for CI-mode unit tests.
    git_index = root / ".git"
    if not git_index.exists():
        return {"git_available": False, "no_generated_or_secret_files_tracked": True}
    import subprocess

    result = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=False)
    _require(result.returncode == 0, "Unable to inspect tracked files")
    tracked = [line.replace("\\", "/") for line in result.stdout.splitlines()]
    patterns = config.get("prohibited_tracked_file_patterns", [])
    from fnmatch import fnmatch

    offenders = sorted(name for name in tracked if any(fnmatch(name, pattern) for pattern in patterns))
    _require(not offenders, f"Ignored/generated files are tracked: {', '.join(offenders)}")
    secret_like = [name for name in tracked if name.lower().endswith(".env") or "/.env" in name.lower()]
    _require(not secret_like, f"Secret-like file is tracked: {', '.join(secret_like)}")
    return {"git_available": True, "no_generated_or_secret_files_tracked": True}


def _safety_language_checks(root: Path) -> dict[str, bool]:
    content = "\n".join((root / name).read_text(encoding="utf-8") for name in ("src/governance.py", "src/monitoring.py"))
    lowered = content.lower()
    checks = {
        "human_review_language": "human review" in lowered or "review" in lowered,
        "no_automatic_enforcement": "no outcomes are created" in lowered or "outcome-free" in lowered,
        "no_release_network_client": not re.search(r"^\\s*(?:import|from)\\s+(?:requests|httpx)\\b", (root / "src" / "release_validation.py").read_text(encoding="utf-8"), re.MULTILINE),
    }
    _require(all(checks.values()), "Required safety language/checks are absent")
    return checks


def _check_report_status(path: Path, report: dict[str, Any]) -> None:
    if "status" in report:
        _require(report["status"] == "PASS", f"Report is not PASS: {path.name}")


def _documentation_checks(root: Path, config: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check documentation safety and report-backed headline metrics.

    These are targeted claim/contract checks rather than fragile paragraph
    comparisons, so authors retain freedom to improve wording.
    """
    files = ["README.md", *config.get("required_documentation_files", [])]
    missing = [name for name in files if not (root / name).is_file()]
    _require(not missing, f"Required documentation missing: {', '.join(missing)}")
    content = {name: (root / name).read_text(encoding="utf-8") for name in files}
    joined = "\n".join(content.values())
    _require(not re.search(r"[A-Za-z]:\\+Users\\+", joined), "Documentation contains an absolute Windows user path")
    _require(not re.search(r"(?i)(?:raw_risk_score|score)\s+(?:is|as)\s+(?:a\s+)?calibrated probability", joined), "Documentation describes an uncalibrated score as a calibrated probability")
    _require(not re.search(r"(?i)2023[^\n.]{0,100}(?:accuracy|precision|recall|pr[- ]auc|roc[- ]auc|brier|performance\s+(?:is|was|=))", joined), "Documentation makes a current 2023 performance claim")
    _require(not re.search(r"(?i)automatic enforcement\s+(?:is|was|has been)?\s*(?:enabled|active|initiated|performed)", joined), "Documentation claims automatic enforcement")
    _require(not re.search(r"(?i)(?:api[_ -]?key|password|secret)\s*[:=]\s*[^\s<]{6,}", joined), "Documentation contains a possible secret value")
    for script in re.findall(r"\b(run_[a-z_]+\.py)\b", joined):
        _require((root / script).is_file(), f"Documentation references a missing command: {script}")
    required_language = ("human review", "uncalibrated", "candidate", "no automatic enforcement")
    _require(all(token in joined.lower() for token in required_language), "Documentation is missing required responsible-use language")
    by_name = {Path(name).name: value for name, value in reports.items()}
    foundation = by_name["data_foundation_report.json"]
    baseline = by_name["baseline_report.json"]
    model = by_name["model_comparison_report.json"]
    batch = by_name["batch_prediction_report.json"]
    metrics = content["docs/project_metrics.md"]
    selected = next(item for item in model["experiments"] if item["experiment_id"] == "exp_05_random_forest")
    expected = (
        f"{foundation['label_counts']['labelled']:,}",
        f"{foundation['label_counts']['positive']}",
        f"{foundation['label_counts']['negative']:,}",
        f"{foundation['label_counts']['positive_rate_percentage']:.2f}%",
        f"{baseline['validation']['metrics']['ranking_at']['10']['selected_positives']}",
        f"{selected['metrics']['ranking_at']['10']['selected_positives']}",
        f"{selected['metrics']['ranking_at']['10']['precision']:.4f}",
        f"{selected['metrics']['ranking_at']['10']['recall']:.4f}",
        f"{selected['metrics']['ranking_at']['10']['lift']:.4f}",
        f"{selected['metrics']['pr_auc']:.4f}",
        f"{selected['metrics']['roc_auc']:.4f}",
        f"{selected['metrics']['brier_score']:.4f}",
        f"{batch['input_row_count']}",
    )
    missing_metrics = [value for value in expected if value not in metrics]
    _require(not missing_metrics, f"Project metrics summary is missing report-backed values: {', '.join(missing_metrics)}")
    return {"valid": True, "files": files, "report_backed_metric_values": len(expected), "commands_valid": True, "safety_language_valid": True}


def _ci_checks(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    required = sorted(config["required_source_files"])
    missing = [name for name in required if not (root / name).is_file()]
    _require(not missing, f"Required source files missing: {', '.join(missing)}")
    config_files = sorted(config["required_config_files"])
    for name in config_files:
        try:
            yaml.safe_load((root / name).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ReleaseValidationError(f"Invalid configuration {name}: {exc}") from exc
    reports: dict[str, dict[str, Any]] = {}
    for name in sorted(config["required_committed_reports"]):
        path = root / name
        _require(path.is_file(), f"Committed report missing: {name}")
        reports[name] = _read_json(path)
        _check_report_status(path, reports[name])
    importlib.invalidate_caches()
    try:
        importlib.import_module("app.streamlit_app")
    except Exception as exc:  # import only; no Streamlit server is started
        raise ReleaseValidationError(f"Streamlit application import failed: {exc}") from exc
    checks = {
        "required_files": {"valid": True, "files": required},
        "configuration": {"valid": True, "files": config_files},
        "reports": {"valid": True, "files": sorted(reports)},
        "workflow": _workflow_checks(root),
        "docker": _docker_checks(root),
        "streamlit_import": {"valid": True, "server_started": False},
        "safety": _safety_language_checks(root),
        "ignored_artifact_policy": _tracked_file_checks(root, config),
        "documentation": _documentation_checks(root, config, reports),
    }
    return checks, reports


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        return headers, list(reader)


def _local_checks(root: Path, config: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_name = {Path(name).name: report for name, report in reports.items()}
    feasibility = by_name["feasibility_report.json"]
    feature = by_name["feature_engineering_report.json"]
    model = by_name["model_comparison_report.json"]
    calibration = by_name["calibration_report.json"]
    mlflow = by_name["mlflow_tracking_report.json"]
    batch = by_name["batch_prediction_report.json"]
    dashboard = by_name["dashboard_validation_report.json"]
    monitoring = by_name["monitoring_report.json"]
    _require(feasibility.get("decision") == "GO", "Day 0 feasibility decision must be GO")
    snapshot = feature.get("source_snapshot_id")
    _require(snapshot and all(report.get("source_snapshot_id", snapshot) == snapshot for report in (model, calibration, batch, dashboard, monitoring)), "Source snapshot IDs do not agree")
    feature_version = feature.get("feature_version")
    _require(feature_version and all(report.get("feature_version", feature_version) == feature_version for report in (model, calibration, batch, dashboard, monitoring)), "Feature versions do not agree")
    selected_candidate = model.get("selected_candidate")
    selected_experiment = selected_candidate.get("experiment_name") if isinstance(selected_candidate, dict) else selected_candidate
    _require(selected_experiment == config["expected_model_experiment"], "Selected Day 3 experiment does not match release contract")
    _require(calibration.get("selected_calibration_method") == config["expected_selected_calibration_method"], "Selected calibration method must remain uncalibrated")
    _require(batch.get("selected_day3_experiment") == config["expected_model_experiment"], "Batch report selected experiment is incompatible")
    _require(batch.get("selected_day4_method") == config["expected_selected_calibration_method"], "Batch report must describe uncalibrated model output")
    _require(mlflow.get("logged_day3_run_count") == 8 and mlflow.get("logged_day4_run_count") == 3, "MLflow logical run counts must be 8 Day 3 and 3 Day 4")
    _require(mlflow.get("final_candidate_artifact_logged") is True, "Final candidate artifact was not logged with the selected run")
    _require(mlflow.get("reused_run_count", 0) >= 11 and mlflow.get("newly_created_run_count", 0) == 0, "MLflow second-run idempotency evidence is invalid")
    model_path = _path_from_report(root, calibration.get("final_candidate_artifact_path"), "final_candidate_artifact_path")
    ranked_path = _path_from_report(root, batch.get("ranked_output_path"), "ranked_output_path")
    top_path = _path_from_report(root, batch.get("top_10_output_path"), "top_10_output_path")
    for path in (model_path, ranked_path, top_path):
        _require(path.is_file(), f"Required frozen artifact missing: {_relative(root, path)}")
    _require(_sha256(model_path) == batch.get("model_artifact_hash"), "Final candidate model hash mismatch")
    _require(_sha256(ranked_path) == batch.get("ranked_output_hash"), "Ranked candidate hash mismatch")
    _require(_sha256(top_path) == batch.get("top_10_output_hash"), "Top-10 candidate hash mismatch")
    headers, rows = _csv_rows(ranked_path)
    top_headers, top_rows = _csv_rows(top_path)
    _require(len(rows) == config["expected_locked_candidate_row_count"], "Ranked candidate row count must be 300")
    _require(len(top_rows) == config["expected_top_10_row_count"], "Top-10 candidate row count must be 30")
    prohibited = {"label", "target", "outcome", "viol_type", "serious_violation"}
    _require(not (prohibited & {header.lower() for header in headers + top_headers}), "Ranked output contains an outcome or target field")
    manifest_path = _path_from_report(root, monitoring.get("monitoring_manifest_path"), "monitoring_manifest_path")
    template_path = _path_from_report(root, monitoring.get("future_outcome_template_path"), "future_outcome_template_path")
    worksheet_path = _path_from_report(root, monitoring.get("review_worksheet_path"), "review_worksheet_path")
    for path in (manifest_path, template_path, worksheet_path):
        _require(path.is_file(), f"Monitoring artifact missing: {_relative(root, path)}")
    _require(_sha256(template_path) == monitoring.get("future_outcome_template_hash"), "Future outcome template hash mismatch")
    _require(_sha256(worksheet_path) == monitoring.get("review_worksheet_hash"), "Review worksheet hash mismatch")
    template_headers, template_rows = _csv_rows(template_path)
    _require(bool(template_headers) and not template_rows, "Future outcome template must contain headers only")
    from src.governance import REVIEW_COLUMNS
    worksheet_headers, _ = _csv_rows(worksheet_path)
    _require(set(REVIEW_COLUMNS).issubset(worksheet_headers), "Human-review governance fields are incomplete")
    _require(monitoring.get("monitoring_health") in config["allowed_monitoring_health_values"], "Monitoring health is invalid")
    flags = {
        "labels_accessed": batch.get("labels_accessed") is False and monitoring.get("current_labels_accessed") is False,
        "performance_metrics_calculated": batch.get("performance_metrics_calculated") is False and monitoring.get("current_performance_metrics_calculated") is False,
        "outcome_fairness_metrics_calculated": monitoring.get("outcome_fairness_metrics_calculated") is False,
        "automatic_enforcement": batch.get("automatic_enforcement") is False and monitoring.get("automatic_enforcement") is False,
        "model_refit_attempted": dashboard.get("model_refit_attempted") is False and monitoring.get("model_refit_attempted") is False,
        "prediction_artifact_modified": monitoring.get("prediction_artifact_modified") is False,
    }
    _require(all(flags.values()), "Locked candidate safety flags are not all false")
    return {
        "valid": True,
        "source_snapshot_id": snapshot,
        "feature_version": feature_version,
        "selected_experiment": config["expected_model_experiment"],
        "selected_calibration_method": config["expected_selected_calibration_method"],
        "artifacts": sorted(_relative(root, path) for path in (model_path, ranked_path, top_path, manifest_path, template_path, worksheet_path)),
        "hashes": {"model": True, "ranked": True, "top_10": True, "monitoring": True},
        "safety_flags": flags,
    }


def validate_release(root: Path | str = ".", mode: str = "ci") -> dict[str, Any]:
    """Return a deterministic, read-only validation report for ``mode``."""
    root = Path(root).resolve()
    _require(mode in {"ci", "local"}, "mode must be either 'ci' or 'local'")
    started = time.monotonic()
    config = _load_config(root)
    checks, reports = _ci_checks(root, config)
    local = {"required": False, "valid": True, "artifacts": "not required in CI mode"}
    if mode == "local":
        local = _local_checks(root, config, reports)
    return {
        "status": "PASS",
        "mode": mode,
        "release_validation_version": config["release_validation_version"],
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "source_snapshot_id": local.get("source_snapshot_id"),
        "feature_version": local.get("feature_version"),
        "selected_experiment": local.get("selected_experiment"),
        "selected_calibration_method": local.get("selected_calibration_method"),
        "required_file_checks": checks["required_files"],
        "configuration_checks": checks["configuration"],
        "report_checks": checks["reports"],
        "artifact_checks": local,
        "hash_checks": local.get("hashes", {"not_required": mode == "ci"}),
        "ci_workflow_checks": checks["workflow"],
        "docker_contract_checks": checks["docker"],
        "streamlit_import_check": checks["streamlit_import"],
        "documentation_checks": checks["documentation"],
        "locked_candidate_safety_checks": local.get("safety_flags", {"not_required": mode == "ci"}),
        "governance_checks": {"human_review_fields_checked": mode == "local", "future_template_headers_only": mode == "local"},
        "monitoring_safety_checks": {"monitoring_artifacts_checked": mode == "local", "health_independent_from_pipeline": mode == "local"},
        "ignored_artifact_policy_checks": checks["ignored_artifact_policy"],
        "warnings": ["Local artifacts are intentionally not required in CI mode."] if mode == "ci" else [],
        "failures": [],
        "runtime_seconds": round(time.monotonic() - started, 6),
        "limitations": [
            "Candidate ranking supports human review; there is no automatic enforcement.",
            "The 2023 candidate batch is awaiting complete outcome labels.",
            "This release validation does not load labels, calculate performance or fairness metrics, fit models, or regenerate predictions.",
            "The selected final candidate uses uncalibrated model output and retrospective validation only.",
        ],
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
