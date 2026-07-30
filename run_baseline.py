"""Run the cache-only chronological NAICS rate baseline for InspectIQ Day 1."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.baseline import DEFAULT_ALPHA, DEFAULT_MIN_GROUP_SIZE, build_industry_rates, choose_naics_grouping, score_validation_rows
from src.data_foundation import FoundationError, read_json, sha256_file, write_csv, write_json
from src.path_utils import RelativePathError, resolve_report_path
from src.ranking_metrics import validation_metrics
from src.splitting import SplitError, create_chronological_split, write_split_artifacts
from src.validation import load_schema, validate_output_columns


class BaselineError(RuntimeError):
    """The local processed snapshot cannot produce a valid baseline result."""


def read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise BaselineError(f"Could not read labelled snapshot: {error}") from error


def verify_processed_snapshot(foundation: dict[str, Any], base_directory: Path) -> tuple[Path, list[dict[str, Any]]]:
    output = foundation.get("processed_output", {})
    try:
        directory = resolve_report_path(output.get("path", ""), base_directory)
    except RelativePathError as error:
        raise BaselineError(str(error)) from error
    labelled_path = directory / "labelled_inspections.csv"
    expected = output.get("hashes", {}).get("labelled_inspections.csv")
    if not directory.exists() or not expected or not labelled_path.exists() or sha256_file(labelled_path) != expected:
        raise BaselineError("The labelled processed snapshot is unavailable or fails its recorded hash verification.")
    rows = read_csv(labelled_path)
    if not rows:
        raise BaselineError("The labelled processed snapshot is empty.")
    return directory, rows


def run_baseline(*, foundation_report_path: Path, schema_path: Path, artifact_root: Path | None = None) -> dict[str, Any]:
    foundation = read_json(foundation_report_path)
    schema = load_schema(schema_path)
    processed_directory, rows = verify_processed_snapshot(foundation, foundation_report_path.parent.parent)
    validate_output_columns(list(rows[0]), schema)
    try:
        split = create_chronological_split(rows)
    except SplitError as error:
        raise BaselineError(str(error)) from error
    artifact_directory = (artifact_root or processed_directory / "baseline")
    split_manifest = write_split_artifacts(artifact_directory / "splits", split)

    digits, grouping = choose_naics_grouping(split["rows"]["train"], DEFAULT_MIN_GROUP_SIZE)
    rates = build_industry_rates(split["rows"]["train"], digits, DEFAULT_ALPHA)
    scored = score_validation_rows(split["rows"]["validation"], rates, DEFAULT_MIN_GROUP_SIZE)
    scoring_name = "validation_scored.csv"
    write_csv(artifact_directory / scoring_name, scored, ["rank", "activity_nr", "open_date", "industry_group", "baseline_score", "actual_label", "score_source"])
    metrics = validation_metrics(scored)

    return {
        "status": "PASS", "source_snapshot_id": foundation.get("snapshot_id"), "source": foundation.get("source"),
        "input": {"labelled_row_count": len(rows), "labelled_hash": sha256_file(processed_directory / "labelled_inspections.csv")},
        "chronological_split": {key: value for key, value in split.items() if key != "rows"},
        "split_artifacts": {"directory": str(artifact_directory / "splits"), "manifest": split_manifest},
        "baseline": {
            "method": "NAICS historical serious/willful/repeat rate ranking", "training_only": True,
            "naics_grouping": grouping, "smoothing": {"formula": "(positive_count + alpha * global_positive_rate) / (row_count + alpha)", "alpha": DEFAULT_ALPHA},
            "minimum_group_size": DEFAULT_MIN_GROUP_SIZE, "global_positive_rate": rates["global_positive_rate"], "training_group_rates": rates["groups"],
        },
        "validation": {"scoring_output": str(artifact_directory / scoring_name), "metrics": metrics},
        "locked_test": {"row_count": split["periods"]["test"]["row_count"], "open_date_range": split["periods"]["test"]["open_date_range"], "metrics_calculated": False},
        "limitations": [
            "Metrics describe validation only; the locked test set was not scored or used for baseline selection.",
            "Industry rates are estimated exclusively from training labels and use transparent smoothing.",
            "Missing, unseen, and sparse NAICS groups use the training global positive-rate fallback.",
            "This is a descriptive ranking baseline, not a calibrated probability model or an operational decision rule.",
        ],
    }


def main() -> None:
    try:
        report = run_baseline(foundation_report_path=Path("reports/data_foundation_report.json"), schema_path=Path("config/schema.yaml"))
        write_json(Path("reports/baseline_report.json"), report)
    except (BaselineError, FoundationError, OSError, ValueError) as error:
        write_json(Path("reports/baseline_attempt_error.json"), {"status": "FAIL", "error": str(error)})
        print(f"INSPECTIQ BASELINE ERROR: {error}")
        print("INSPECTIQ BASELINE: FAIL")
        return
    metrics = report["validation"]["metrics"]
    print(f"split={report['chronological_split']['strategy']} train={report['chronological_split']['periods']['train']['row_count']} validation={metrics['row_count']} locked_test={report['locked_test']['row_count']}")
    print(f"naics_digits={report['baseline']['naics_grouping']['selected_digits']} validation_pr_auc={metrics['pr_auc']} validation_roc_auc={metrics['roc_auc']}")
    print("INSPECTIQ BASELINE: PASS")


if __name__ == "__main__":
    main()
