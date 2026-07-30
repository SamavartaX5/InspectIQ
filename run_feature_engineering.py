"""Create cache-only Day 2 historical feature artifacts for the locked baseline split."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.feature_engineering import (
    HISTORICAL_FEATURES,
    PROHIBITED_FEATURE_NAMES,
    TARGET_COLUMN,
    FeatureEngineeringError,
    build_training_features,
    employee_count_assessment,
    feature_columns,
    feature_metadata,
    inspect_establishment_key,
    load_feature_config,
    missing_percentages,
    numeric_ranges,
    ordered_rows,
    transform_batch,
    write_feature_artifacts,
)
from src.path_utils import RelativePathError, resolve_report_path


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureEngineeringError(f"Could not read {path}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise FeatureEngineeringError(f"Could not read split input {path}: {error}") from error


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def load_verified_splits(baseline: dict[str, Any], base_directory: Path) -> tuple[Path, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    artifact = baseline.get("split_artifacts", {})
    try:
        directory = resolve_report_path(artifact.get("directory", ""), base_directory)
    except RelativePathError as error:
        raise FeatureEngineeringError(str(error)) from error
    manifest_path = directory / "split_manifest.json"
    manifest = read_json(manifest_path)
    if not manifest.get("strictly_ordered") or manifest.get("id_overlap_count") != 0:
        raise FeatureEngineeringError("Split manifest does not establish a strict, disjoint chronological split.")
    hashes = manifest.get("file_hashes", {})
    expected_names = {"train": "train.csv", "validation": "validation.csv", "test": "test_locked.csv"}
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, name in expected_names.items():
        path = directory / name
        if name not in hashes or not path.exists() or sha256_file(path) != hashes[name]:
            raise FeatureEngineeringError(f"Split input fails its manifest hash verification: {path}")
        rows[split] = read_csv(path)
    if not all(rows.values()):
        raise FeatureEngineeringError("Every chronological split must have at least one row.")
    if TARGET_COLUMN not in rows["train"][0] or TARGET_COLUMN not in rows["validation"][0]:
        raise FeatureEngineeringError("Training and validation split files must include the completed target.")
    if TARGET_COLUMN in rows["test"][0]:
        raise FeatureEngineeringError("The locked test split must not contain the target column.")
    ids = {split: {str(row.get("activity_nr", "")) for row in values} for split, values in rows.items()}
    if any(not identifier for values in ids.values() for identifier in values) or any(len(ids[name]) != len(rows[name]) for name in rows):
        raise FeatureEngineeringError("Split inputs require unique non-empty activity_nr values.")
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise FeatureEngineeringError("Split input activity_nr values overlap.")
    return directory, rows, manifest


def date_range(rows: list[dict[str, Any]]) -> dict[str, str]:
    dates = [str(row["open_date"])[:10] for row in rows]
    return {"min": min(dates), "max": max(dates)}


def report_split_stats(rows: list[dict[str, Any]], output: list[dict[str, Any]], generated_columns: list[str]) -> dict[str, Any]:
    cold_start_count = sum(row["industry_prior_inspection_count"] == 0 for row in output)
    missing = missing_percentages(output, generated_columns)
    return {
        "input_row_count": len(rows), "output_row_count": len(output), "output_column_count": len(output[0]) if output else 0,
        "open_date_range": date_range(rows), "target_present": TARGET_COLUMN in output[0],
        "missing_value_percentages": missing, "feature_coverage_percentages": {name: round(100 - value, 2) for name, value in missing.items()},
        "cold_start_percentage": round(100 * cold_start_count / len(output), 2),
        "industry_history_status_counts": dict(Counter(row["industry_history_status"] for row in output)),
    }


def run_feature_engineering(
    *, foundation_report_path: Path, baseline_report_path: Path, config_path: Path, artifact_root: Path | None = None
) -> dict[str, Any]:
    started = time.perf_counter()
    foundation = read_json(foundation_report_path)
    baseline = read_json(baseline_report_path)
    config = load_feature_config(config_path)
    snapshot_id = foundation.get("snapshot_id")
    if not snapshot_id or baseline.get("source_snapshot_id") != snapshot_id:
        raise FeatureEngineeringError("Data-foundation and baseline reports do not identify the same source snapshot.")
    split_directory, inputs, split_manifest = load_verified_splits(baseline, baseline_report_path.parent.parent)
    all_input_rows = [*inputs["train"], *inputs["validation"], *inputs["test"]]
    key_assessment = inspect_establishment_key(all_input_rows)
    employee = employee_count_assessment(all_input_rows, float(config["employee_count"]["minimum_coverage_percentage"]))
    include_employee = bool(employee["included"])

    training_features = build_training_features(inputs["train"], config, include_employee=include_employee)
    validation_features = transform_batch(inputs["train"], inputs["validation"], config, include_target=True, include_employee=include_employee)
    # The locked test file has no target; only completed train and validation labels form history.
    test_features = transform_batch([*inputs["train"], *inputs["validation"]], inputs["test"], config, include_target=False, include_employee=include_employee)
    outputs = {"train_features.csv": (training_features, feature_columns(True, include_employee)), "validation_features.csv": (validation_features, feature_columns(True, include_employee)), "test_locked_features.csv": (test_features, feature_columns(False, include_employee))}
    output_columns = {column for _name, (_rows, columns) in outputs.items() for column in columns}
    prohibited = sorted(output_columns & PROHIBITED_FEATURE_NAMES)
    if prohibited or TARGET_COLUMN in outputs["test_locked_features.csv"][1]:
        raise FeatureEngineeringError(f"Feature output includes a prohibited field: {prohibited}")
    output_directory = artifact_root or Path(foundation["processed_output"]["path"]) / "features" / config["feature_version"]
    manifest_input = {
        "source_snapshot_id": snapshot_id, "feature_version": config["feature_version"],
        "split_manifest_path": str(split_directory / "split_manifest.json"), "split_manifest_hashes": split_manifest["file_hashes"],
        "output_row_counts": {"train": len(training_features), "validation": len(validation_features), "test_locked": len(test_features)},
        "output_columns": {"train": outputs["train_features.csv"][1], "validation": outputs["validation_features.csv"][1], "test_locked": outputs["test_locked_features.csv"][1]},
    }
    artifact_manifest, reused = write_feature_artifacts(output_directory, outputs, manifest_input)
    generated_columns = [item["feature_name"] for item in feature_metadata(include_employee)]
    numeric_historical = HISTORICAL_FEATURES[:3]
    result = {
        "status": "PASS", "source_snapshot_id": snapshot_id, "feature_version": config["feature_version"],
        "split_input_paths": {"train": str(split_directory / "train.csv"), "validation": str(split_directory / "validation.csv"), "test_locked": str(split_directory / "test_locked.csv")},
        "output_paths": {"directory": str(output_directory), "train": str(output_directory / "train_features.csv"), "validation": str(output_directory / "validation_features.csv"), "test_locked": str(output_directory / "test_locked_features.csv"), "manifest": str(output_directory / "feature_manifest.json")},
        "artifacts_reused": reused,
        "splits": {"train": report_split_stats(inputs["train"], training_features, generated_columns), "validation": report_split_stats(inputs["validation"], validation_features, generated_columns), "test_locked": report_split_stats(inputs["test"], test_features, generated_columns)},
        "establishment_key": key_assessment,
        "naics": {"grouping": "2-digit", "coverage_percentage": {name: round(100 * sum(bool(str(row.get("naics_code") or "").strip()) for row in values) / len(values), 2) for name, values in inputs.items()}, "reason": "Reuses the measured 2-digit NAICS grouping selected by the Day 1 baseline."},
        "nr_in_estab": employee,
        "generated_features": generated_columns,
        "feature_metadata": feature_metadata(include_employee),
        "excluded_candidate_fields": {
            "establishment_history": key_assessment["decision"], "sic_code": "100% missing in the Day 1 validated dataset.",
            "site_state_or_state_flag": "All processed observations are CA, so this feature is constant.",
            "host_est_key_and_name_address_zip": "Not present in the validated split input schema.",
            "violation_and_case_fields": "Prohibited because they reveal inspection outcomes or post-inspection events.",
        },
        "numeric_historical_feature_ranges": {"train": numeric_ranges(training_features, numeric_historical), "validation": numeric_ranges(validation_features, numeric_historical), "test_locked": numeric_ranges(test_features, numeric_historical)},
        "leakage_checks": {
            "history_requires_open_date_strictly_less_than_candidate": True, "current_candidate_label_not_used_for_history": True,
            "candidate_batch_labels_not_used": True, "validation_history_is_training_only": True,
            "locked_test_history_is_training_plus_validation_only": True, "locked_test_target_absent": TARGET_COLUMN not in test_features[0],
            "prohibited_feature_columns_absent": not prohibited,
        },
        "same_day_history_checks": {"training_rows_processed_as_date_batches": True, "same_date_labels_not_visible_within_batch": True},
        "output_hashes": artifact_manifest["file_hashes"], "pipeline_runtime_seconds": round(time.perf_counter() - started, 4),
        "limitations": [
            "Exact establishment matching may miss changes in names or addresses.",
            "No fuzzy establishment matching is used.",
            "Historical OSHA inspections are selection-biased.",
            "Historical features describe only represented inspection history.",
            "Candidate batches receive no outcome information from other rows in the same batch.",
            "Locked-test labels remain unused.",
        ],
    }
    return result


def main() -> None:
    try:
        report = run_feature_engineering(
            foundation_report_path=Path("reports/data_foundation_report.json"), baseline_report_path=Path("reports/baseline_report.json"),
            config_path=Path("config/feature_config.yaml"),
        )
        write_json_atomic(Path("reports/feature_engineering_report.json"), report)
    except (FeatureEngineeringError, OSError, ValueError) as error:
        write_json_atomic(Path("reports/feature_engineering_attempt_error.json"), {"status": "FAIL", "error": str(error)})
        print(f"INSPECTIQ FEATURE ENGINEERING ERROR: {error}")
        print("INSPECTIQ FEATURE ENGINEERING: FAIL")
        return
    print(f"establishment_key={report['establishment_key']['selected_strategy']} train={report['splits']['train']['output_row_count']} validation={report['splits']['validation']['output_row_count']} test_locked={report['splits']['test_locked']['output_row_count']}")
    print(f"historical_features={len(HISTORICAL_FEATURES)} establishment_history_coverage={report['establishment_key']['coverage_percentage']}% cold_start_train={report['splits']['train']['cold_start_percentage']}% cold_start_validation={report['splits']['validation']['cold_start_percentage']}% cold_start_test={report['splits']['test_locked']['cold_start_percentage']}%")
    print(f"naics_coverage={report['naics']['coverage_percentage']} locked_test_target_absent={report['leakage_checks']['locked_test_target_absent']} output_directory={report['output_paths']['directory']}")
    print("INSPECTIQ FEATURE ENGINEERING: PASS")


if __name__ == "__main__":
    main()
