"""Deterministic cache-only snapshot and labelled-table pipeline for Day 1."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_feasibility import NON_POSITIVE_TYPES, POSITIVE_TYPES, build_label_table
from src.validation import ValidationError, load_schema, missing_percentages, validate_inspections, validate_output_columns


LABEL_MAPPING_VERSION = "day0-viol-type-v1"
LABEL_MAPPING = {"S": "Serious", "W": "Willful", "R": "Repeat", "O": "Other", "U": "Unclassified"}
SOURCE_NAME = "Official U.S. Department of Labor / OSHA cache"
ENDPOINTS = ["inspection", "violation"]
CODE_VERSION = "day1-data-foundation-v1"


class FoundationError(RuntimeError):
    """The offline foundation cannot produce a valid immutable artifact."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise FoundationError(f"Could not read {path}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_snapshot_id(configuration: dict[str, Any], inspection_ids: list[str], completed_ids: list[str]) -> str:
    material = {
        "state": configuration["state"], "start_date": configuration["start_date"],
        "end_date": configuration["end_date"], "inspection_activity_ids": sorted(inspection_ids),
        "completed_violation_activity_ids": sorted(completed_ids),
        "label_mapping_version": LABEL_MAPPING_VERSION,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def csv_columns(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    available = {key for row in rows for key in row}
    ordered = [name for name in preferred or [] if name in available]
    return ordered + sorted(available - set(ordered))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def load_page_rows(directory: Path, manifest: dict[str, Any], wanted_rows: int | None = None) -> list[dict[str, Any]]:
    pages = manifest.get("pages")
    if not manifest.get("complete") or not isinstance(pages, dict):
        raise FoundationError(f"Cache manifest is incomplete: {directory / 'manifest.json'}")
    rows: list[dict[str, Any]] = []
    for offset in sorted((int(key) for key in pages if key.isdigit())):
        page = pages[str(offset)]
        if page.get("status") != "success" or not isinstance(page.get("file"), str):
            raise FoundationError(f"Cache page is not successful: {directory / 'manifest.json'} offset {offset}")
        page_rows = read_json(directory / page["file"])
        if not isinstance(page_rows, list) or page.get("row_count") != len(page_rows):
            raise FoundationError(f"Cache page hash/row contract is invalid: {directory / page['file']}")
        rows.extend(row for row in page_rows if isinstance(row, dict))
    return rows[:wanted_rows] if wanted_rows is not None else rows


def load_day0_cache(cache_root: Path, configuration: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], list[str]]:
    inspection_root = cache_root / "inspection"
    if not inspection_root.exists():
        raise FoundationError(f"Required inspection cache is unavailable: {inspection_root}")
    inspections: list[dict[str, Any]] = []
    sources: list[str] = []
    for year in range(int(configuration["start_date"][:4]), int(configuration["end_date"][:4]) + 1):
        directory = inspection_root / f"year_{year}"
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FoundationError(f"Required cached inspection year is unavailable: {manifest_path}")
        manifest = read_json(manifest_path)
        request = manifest.get("request", {})
        if request.get("endpoint") != "inspection":
            raise FoundationError(f"Inspection manifest has an unexpected endpoint: {manifest_path}")
        inspections.extend(load_page_rows(directory, manifest, request.get("wanted_rows")))
        sources.append(str(manifest_path))
    violations: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    violation_root = cache_root / "violation"
    if not violation_root.exists():
        raise FoundationError(f"Required violation cache is unavailable: {violation_root}")
    seen_rows: set[tuple[str, str]] = set()
    # Day 0 deliberately reused completed batches across compatible audits;
    # preserve that provenance instead of narrowing the snapshot to one folder.
    for manifest_path in sorted(cache_root.parent.glob("audit_*/violation/batch_*/manifest.json")):
        manifest = read_json(manifest_path)
        request = manifest.get("request", {})
        filters = request.get("filters", {})
        batch_ids = {str(value) for value in filters.get("value", [])}
        if request.get("endpoint") != "violation" or not manifest.get("complete"):
            continue
        rows = load_page_rows(manifest_path.parent, manifest)
        completed_ids.update(batch_ids)
        for row in rows:
            key = (str(row.get("activity_nr")), str(row.get("citation_id")))
            if key not in seen_rows:
                seen_rows.add(key)
                violations.append(row)
        sources.append(str(manifest_path))
    inspection_ids = {str(row.get("activity_nr")) for row in inspections if row.get("activity_nr") is not None}
    completed_ids &= inspection_ids
    violations = [row for row in violations if str(row.get("activity_nr")) in inspection_ids]
    if not inspections or not completed_ids:
        raise FoundationError("Cached Day 0 data is unavailable or contains no completed violation retrieval IDs.")
    return inspections, violations, completed_ids, sources


def verify_immutable(directory: Path, manifest_name: str) -> dict[str, Any]:
    manifest_path = directory / manifest_name
    if not manifest_path.exists():
        raise FoundationError(f"Immutable directory exists without {manifest_name}: {directory}")
    manifest = read_json(manifest_path)
    for name, expected in manifest.get("file_hashes", {}).items():
        path = directory / name
        if not path.exists() or sha256_file(path) != expected:
            raise FoundationError(f"Immutable file hash verification failed: {path}")
    return manifest


def create_raw_snapshot(
    *, snapshot_root: Path, snapshot_id: str, configuration: dict[str, Any], inspections: list[dict[str, Any]],
    violations: list[dict[str, Any]], completed_ids: set[str], sources: list[str]
) -> tuple[Path, dict[str, Any], bool]:
    directory = snapshot_root / snapshot_id
    if directory.exists():
        manifest = verify_immutable(directory, "manifest.json")
        if manifest.get("snapshot_id") != snapshot_id:
            raise FoundationError(f"Existing snapshot ID does not match directory: {directory}")
        return directory, manifest, True
    directory.mkdir(parents=True, exist_ok=False)
    inspection_columns = csv_columns(inspections)
    violation_columns = csv_columns(violations)
    inspections_path = directory / "inspections.csv"
    violations_path = directory / "violations.csv"
    write_csv(inspections_path, sorted(inspections, key=lambda row: str(row.get("activity_nr"))), inspection_columns)
    write_csv(violations_path, sorted(violations, key=lambda row: (str(row.get("activity_nr")), str(row.get("citation_id")))), violation_columns)
    dates = sorted(str(row.get("open_date")) for row in inspections if row.get("open_date"))
    manifest = {
        "snapshot_id": snapshot_id, "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_name": SOURCE_NAME, "official_endpoints": ENDPOINTS, "state": configuration["state"],
        "requested_date_range": {"start": configuration["start_date"], "end": configuration["end_date"]},
        "actual_open_date_range": {"min": dates[0] if dates else None, "max": dates[-1] if dates else None},
        "inspection_row_count": len(inspections), "violation_row_count": len(violations),
        "completed_violation_retrieval_id_count": len(completed_ids),
        "excluded_incomplete_id_count": len(inspections) - len(completed_ids),
        "label_mapping_version": LABEL_MAPPING_VERSION, "label_mapping": LABEL_MAPPING,
        "source_cache_manifests": sources, "code_version": CODE_VERSION, "schema_version": "day1-data-foundation-v1",
        "file_hashes": {"inspections.csv": sha256_file(inspections_path), "violations.csv": sha256_file(violations_path)},
    }
    write_json(directory / "manifest.json", manifest)
    return directory, manifest, False


def build_processed_tables(
    inspections: list[dict[str, Any]], violations: list[dict[str, Any]], completed_ids: set[str], schema: dict[str, Any], configuration: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted, rejected, validation = validate_inspections(
        inspections, schema, state=configuration["state"], start_date=configuration["start_date"], end_date=configuration["end_date"]
    )
    labels, label_details = build_label_table(accepted, violations, completed_ids)
    labels_by_id = {str(row["activity_nr"]): row for row in labels}
    feature_columns = [item["name"] for item in schema["inspection_fields"] if item.get("feature")]
    labelled: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in accepted:
        outcome = labels_by_id[str(row["activity_nr"])]
        record = {column: row.get(column) for column in feature_columns}
        if outcome["label"] is None:
            excluded.append({**record, "exclusion_reason": outcome["label_exclusion_reason"]})
        else:
            labelled.append({**record, "serious_violation_found": outcome["label"]})
    validate_output_columns(csv_columns(labelled, feature_columns + ["serious_violation_found"]), schema)
    metrics = {
        **validation, **label_details,
        "feature_columns_retained": feature_columns,
        "leakage_fields_excluded": schema["prohibited_leakage_columns"],
        "excluded_reason_counts": dict(Counter(row["exclusion_reason"] for row in excluded)),
    }
    return labelled, excluded, rejected, metrics


def create_processed_output(
    *, processed_root: Path, snapshot_id: str, labelled: list[dict[str, Any]], excluded: list[dict[str, Any]], rejected: list[dict[str, Any]], schema: dict[str, Any]
) -> tuple[Path, dict[str, Any], bool]:
    directory = processed_root / snapshot_id
    if directory.exists():
        return directory, verify_immutable(directory, "manifest.json"), True
    directory.mkdir(parents=True, exist_ok=False)
    feature_columns = [item["name"] for item in schema["inspection_fields"] if item.get("feature")]
    outputs = {
        "labelled_inspections.csv": (labelled, feature_columns + ["serious_violation_found"]),
        "excluded_inspections.csv": (excluded, feature_columns + ["exclusion_reason"]),
        "rejected_inspections.csv": (rejected, csv_columns(rejected) or feature_columns + ["rejection_reason"]),
    }
    for name, (rows, columns) in outputs.items():
        write_csv(directory / name, rows, columns)
    manifest = {
        "snapshot_id": snapshot_id, "code_version": CODE_VERSION, "schema_version": schema["schema_version"],
        "file_hashes": {name: sha256_file(directory / name) for name in outputs},
        "row_counts": {"labelled": len(labelled), "excluded": len(excluded), "rejected": len(rejected)},
    }
    write_json(directory / "manifest.json", manifest)
    return directory, manifest, False


def run_foundation(
    *, report_path: Path, schema_path: Path, snapshot_root: Path, processed_root: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    day0 = read_json(report_path)
    configuration = day0.get("configuration", {})
    cache_directory = day0.get("acquisition", {}).get("cache_directory")
    required_config = {"state", "start_date", "end_date"}
    if not required_config.issubset(configuration) or not cache_directory:
        raise FoundationError("Day 0 feasibility report does not identify a usable cache configuration.")
    schema = load_schema(schema_path)
    inspections, violations, completed_ids, sources = load_day0_cache(Path(cache_directory), configuration)
    inspection_ids = [str(row["activity_nr"]) for row in inspections if row.get("activity_nr") is not None]
    snapshot_id = stable_snapshot_id(configuration, inspection_ids, list(completed_ids))
    raw_directory, raw_manifest, raw_reused = create_raw_snapshot(
        snapshot_root=snapshot_root, snapshot_id=snapshot_id, configuration=configuration,
        inspections=inspections, violations=violations, completed_ids=completed_ids, sources=sources,
    )
    labelled, excluded, rejected, metrics = build_processed_tables(inspections, violations, completed_ids, schema, configuration)
    processed_directory, processed_manifest, processed_reused = create_processed_output(
        processed_root=processed_root, snapshot_id=snapshot_id, labelled=labelled, excluded=excluded, rejected=rejected, schema=schema,
    )
    positives = sum(row["serious_violation_found"] == 1 for row in labelled)
    negatives = sum(row["serious_violation_found"] == 0 for row in labelled)
    by_year: dict[str, int] = defaultdict(int)
    for row in labelled:
        by_year[str(row["open_date"])[:4]] += 1
    report = {
        "snapshot_id": snapshot_id, "source": SOURCE_NAME,
        "raw_inspection_shape": [len(inspections), len(csv_columns(inspections))],
        "raw_violation_shape": [len(violations), len(csv_columns(violations))],
        "labelled_table_shape": [len(labelled), len(csv_columns(labelled))],
        "excluded_table_shape": [len(excluded), len(csv_columns(excluded))],
        "rejected_table_shape": [len(rejected), len(csv_columns(rejected))],
        "duplicate_inspection_count": metrics["duplicate_inspection_count"], "invalid_date_count": metrics["invalid_date_count"],
        "missing_value_percentages": missing_percentages(inspections, [item["name"] for item in schema["inspection_fields"]]),
        "naics_coverage_percentage": round(100 - missing_percentages(inspections, ["naics_code"])["naics_code"], 2),
        "sic_coverage_percentage": round(100 - missing_percentages(inspections, ["sic_code"])["sic_code"], 2),
        "label_counts": {"positive": positives, "negative": negatives, "labelled": len(labelled), "positive_rate_percentage": round(100 * positives / len(labelled), 2) if labelled else None},
        "labelled_row_counts_by_year": dict(sorted(by_year.items())),
        "excluded_counts_and_reasons": {"count": len(excluded), "reasons": metrics["excluded_reason_counts"]},
        "completed_vs_incomplete_retrieval": {
            "completed_id_count": len(completed_ids),
            "incomplete_id_count": len(inspections) - len(completed_ids),
            "incomplete_outcomes_assumed_negative": False,
        },
        "rejected_counts_and_reasons": {"count": len(rejected), "reasons": metrics["rejected_reason_counts"]},
        "unknown_categorical_values": metrics["unknown_categorical_values"], "unknown_violation_category_counts": metrics["unknown_violation_category_counts"],
        "feature_columns_retained": metrics["feature_columns_retained"], "leakage_fields_excluded": metrics["leakage_fields_excluded"],
        "raw_snapshot": {"path": str(raw_directory), "reused": raw_reused, "hashes": raw_manifest["file_hashes"]},
        "processed_output": {"path": str(processed_directory), "reused": processed_reused, "hashes": processed_manifest["file_hashes"]},
        "runtime_seconds": round(time.perf_counter() - started, 3), "code_version": CODE_VERSION,
    }
    return report
