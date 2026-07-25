"""Deterministic, leakage-safe historical inspection features."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TARGET_COLUMN = "serious_violation_found"
HISTORICAL_FEATURES = [
    "industry_prior_inspection_count",
    "industry_prior_positive_count",
    "industry_prior_positive_rate_smoothed",
    "industry_history_status",
]
PROHIBITED_FEATURE_NAMES = {
    "citation_id", "viol_type", "delete_flag", "current_penalty", "initial_penalty", "issuance_date",
    "contest_date", "final_order_date", "close_case_date", "close_conf_date", "case_mod_date",
    "why_no_insp", "violation_count",
}
FEATURE_OUTPUT_FILES = {
    "train": "train_features.csv",
    "validation": "validation_features.csv",
    "test_locked": "test_locked_features.csv",
}


class FeatureEngineeringError(RuntimeError):
    """Feature inputs or deterministic artifacts violate the Day 2 contract."""


def parse_open_date(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise FeatureEngineeringError(f"Invalid open_date: {value!r}") from error


def open_day(value: Any):
    """The source is an inspection date, so all timestamps on one day form one history batch."""
    return parse_open_date(value).date()


def activity_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (open_day(row["open_date"]), activity_sort_key(row["activity_nr"])))


def load_feature_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureEngineeringError(f"Could not read feature configuration {path}: {error}") from error
    needed = {"feature_version", "industry_naics_digits", "historical_rate_alpha", "static_features", "employee_count"}
    if not needed.issubset(config) or int(config["industry_naics_digits"]) != 2 or float(config["historical_rate_alpha"]) <= 0:
        raise FeatureEngineeringError("Feature configuration is incomplete or does not preserve the measured 2-digit NAICS grouping.")
    return config


def naics_group(value: Any, digits: int = 2) -> str | None:
    raw = str(value or "").strip().split(".", 1)[0]
    numeric = "".join(character for character in raw if character.isdigit())
    return numeric[:digits] if len(numeric) >= digits else None


def normalize_exact_component(value: Any) -> str | None:
    """Normalization suitable only for an exact, already-verified composite key."""
    if value is None:
        return None
    normalized = " ".join(str(value).strip().upper().split())
    return normalized or None


def normalized_establishment_composite(row: dict[str, Any]) -> str | None:
    fields = ("establishment_name", "site_address", "site_zip")
    components = [normalize_exact_component(row.get(field)) for field in fields]
    return "|".join(components) if all(components) else None


def inspect_establishment_key(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = {column for row in rows for column in row}
    candidates: list[dict[str, Any]] = []
    if "host_est_key" in columns:
        values = [normalize_exact_component(row.get("host_est_key")) for row in rows]
        present = [value for value in values if value]
        candidates.append({"strategy": "host_est_key", "accepted": bool(present), "coverage_percentage": 100 * len(present) / len(rows), "reason": "Present and non-empty." if present else "Present but empty."})
    else:
        candidates.append({"strategy": "host_est_key", "accepted": False, "coverage_percentage": 0.0, "reason": "Field is absent from the validated split inputs."})
    composite_fields = {"establishment_name", "site_address", "site_zip"}
    if composite_fields.issubset(columns):
        keys = [normalized_establishment_composite(row) for row in rows]
        present = [key for key in keys if key]
        candidates.append({"strategy": "normalized_exact_name_address_zip", "accepted": False, "coverage_percentage": 100 * len(present) / len(rows), "reason": "Candidate is available but not selected unless independently verified stable."})
    else:
        candidates.append({"strategy": "normalized_exact_name_address_zip", "accepted": False, "coverage_percentage": 0.0, "reason": "Required establishment name, address, and ZIP fields are absent."})
    return {
        "selected_strategy": "none", "coverage_percentage": 0.0, "distinct_key_count": 0,
        "repeat_key_count": 0, "repeat_key_row_coverage_percentage": 0.0, "candidates": candidates,
        "decision": "No defensible establishment identifier exists in the validated processed fields; establishment-history features are omitted.",
    }


def employee_count_assessment(rows: list[dict[str, Any]], minimum_coverage: float) -> dict[str, Any]:
    values: list[int] = []
    present = 0
    for row in rows:
        raw = row.get("nr_in_estab")
        if raw is not None and str(raw).strip():
            present += 1
            try:
                value = int(str(raw))
            except ValueError:
                continue
            if value >= 0:
                values.append(value)
    coverage = 100 * len(values) / len(rows) if rows else 0.0
    return {
        "source_field": "nr_in_estab", "present_percentage": 100 * present / len(rows) if rows else 0.0,
        "valid_non_negative_integer_percentage": coverage, "minimum": min(values) if values else None,
        "maximum": max(values) if values else None, "included": coverage >= minimum_coverage,
        "reason": "Included: values are present and parse as non-negative integers at or above configured coverage." if coverage >= minimum_coverage else "Excluded: valid non-negative integer coverage is below the configured threshold.",
    }


def _initial_state() -> dict[str, Any]:
    return {"count": 0, "positive": 0, "industry": defaultdict(lambda: {"count": 0, "positive": 0})}


def _add_history(state: dict[str, Any], row: dict[str, Any]) -> None:
    if TARGET_COLUMN not in row:
        raise FeatureEngineeringError("History rows must include completed binary labels.")
    label = int(row[TARGET_COLUMN])
    if label not in (0, 1):
        raise FeatureEngineeringError("History labels must be binary.")
    state["count"] += 1
    state["positive"] += label
    group = naics_group(row.get("naics_code"))
    if group is not None:
        state["industry"][group]["count"] += 1
        state["industry"][group]["positive"] += label


def _history_features(state: dict[str, Any], row: dict[str, Any], config: dict[str, Any], include_employee: bool) -> dict[str, Any]:
    group = naics_group(row.get("naics_code"), int(config["industry_naics_digits"]))
    global_rate = state["positive"] / state["count"] if state["count"] else 0.0
    group_history = state["industry"].get(group) if group is not None else None
    prior_count = group_history["count"] if group_history else 0
    prior_positive = group_history["positive"] if group_history else 0
    alpha = float(config["historical_rate_alpha"])
    if group is None:
        status = "missing_industry_fallback"
    elif group_history is None:
        status = "cold_start_global_fallback" if not state["count"] else "unseen_industry_fallback"
    else:
        status = "industry_history"
    values: dict[str, Any] = {
        "activity_nr": str(row["activity_nr"]), "open_date": str(row["open_date"]),
        "naics_group": group or "", "insp_type": str(row.get("insp_type") or ""),
        "insp_scope": str(row.get("insp_scope") or ""), "owner_type": str(row.get("owner_type") or ""),
        "safety_hlth": str(row.get("safety_hlth") or ""), "open_month": parse_open_date(row["open_date"]).month,
        "industry_prior_inspection_count": prior_count, "industry_prior_positive_count": prior_positive,
        "industry_prior_positive_rate_smoothed": (prior_positive + alpha * global_rate) / (prior_count + alpha),
        "industry_history_status": status,
    }
    if include_employee:
        values["nr_in_estab"] = int(str(row.get("nr_in_estab"))) if str(row.get("nr_in_estab") or "").strip() else None
    return values


def build_training_features(training_rows: list[dict[str, Any]], config: dict[str, Any], *, include_employee: bool = True) -> list[dict[str, Any]]:
    """Build training features date-by-date so same-day outcomes are never visible."""
    state = _initial_state()
    output: list[dict[str, Any]] = []
    rows = ordered_rows(training_rows)
    index = 0
    while index < len(rows):
        date = open_day(rows[index]["open_date"])
        end = index
        while end < len(rows) and open_day(rows[end]["open_date"]) == date:
            end += 1
        same_day = rows[index:end]
        for row in same_day:
            feature_row = _history_features(state, row, config, include_employee)
            feature_row[TARGET_COLUMN] = int(row[TARGET_COLUMN])
            output.append(feature_row)
        for row in same_day:
            _add_history(state, row)
        index = end
    return output


def transform_batch(history_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], config: dict[str, Any], *, include_target: bool = False, include_employee: bool = True) -> list[dict[str, Any]]:
    """Transform a candidate batch using only history records strictly earlier than each candidate date."""
    state = _initial_state()
    history = ordered_rows(history_rows)
    candidates = ordered_rows(candidate_rows)
    output: list[dict[str, Any]] = []
    history_index = 0
    for candidate in candidates:
        candidate_date = open_day(candidate["open_date"])
        while history_index < len(history) and open_day(history[history_index]["open_date"]) < candidate_date:
            _add_history(state, history[history_index])
            history_index += 1
        feature_row = _history_features(state, candidate, config, include_employee)
        if include_target:
            if TARGET_COLUMN not in candidate:
                raise FeatureEngineeringError("Requested target output, but candidate labels are unavailable.")
            feature_row[TARGET_COLUMN] = int(candidate[TARGET_COLUMN])
        output.append(feature_row)
    return output


def feature_columns(include_target: bool, include_employee: bool) -> list[str]:
    columns = ["activity_nr", "open_date", "naics_group", "insp_type", "insp_scope", "owner_type", "safety_hlth"]
    if include_employee:
        columns.append("nr_in_estab")
    columns.extend(["open_month", *HISTORICAL_FEATURES])
    if include_target:
        columns.append(TARGET_COLUMN)
    return columns


def feature_metadata(include_employee: bool) -> list[dict[str, Any]]:
    static = [
        ("naics_group", ["naics_code"], "categorical", "Two-digit NAICS is known before inspection."),
        ("insp_type", ["insp_type"], "categorical", "Inspection type is a pre-inspection source field."),
        ("insp_scope", ["insp_scope"], "categorical", "Inspection scope is a pre-inspection source field."),
        ("owner_type", ["owner_type"], "categorical", "Owner type is a pre-inspection source field."),
        ("safety_hlth", ["safety_hlth"], "categorical", "Safety/health indicator is a pre-inspection source field."),
        ("open_month", ["open_date"], "integer", "Calendar month is derived from the candidate open date."),
    ]
    if include_employee:
        static.insert(5, ("nr_in_estab", ["nr_in_estab"], "integer", "Reported establishment size is available with the candidate record."))
    metadata = [{"feature_name": name, "source_fields": sources, "type": kind, "timing_justification": timing, "missing_value_policy": "Retain empty categorical value or missing numeric value.", "cold_start_policy": "Not applicable.", "leakage_classification": "static_pre_inspection", "available_during_candidate_scoring": True} for name, sources, kind, timing in static]
    metadata.extend([
        {"feature_name": "industry_prior_inspection_count", "source_fields": ["open_date", "naics_code", TARGET_COLUMN], "type": "integer", "timing_justification": "Counts only labelled history with open_date strictly earlier than the candidate.", "missing_value_policy": "Zero for no prior industry history.", "cold_start_policy": "0", "leakage_classification": "strict_prior_history", "available_during_candidate_scoring": True},
        {"feature_name": "industry_prior_positive_count", "source_fields": ["open_date", "naics_code", TARGET_COLUMN], "type": "integer", "timing_justification": "Counts positive labels only from strictly earlier history.", "missing_value_policy": "Zero for no prior industry history.", "cold_start_policy": "0", "leakage_classification": "strict_prior_history", "available_during_candidate_scoring": True},
        {"feature_name": "industry_prior_positive_rate_smoothed", "source_fields": ["open_date", "naics_code", TARGET_COLUMN], "type": "float", "timing_justification": "Uses only the prior industry and global history available before the candidate date.", "missing_value_policy": "Smoothed global prior rate fallback.", "cold_start_policy": "0.0 when no prior labelled history exists.", "leakage_classification": "strict_prior_history", "available_during_candidate_scoring": True},
        {"feature_name": "industry_history_status", "source_fields": ["open_date", "naics_code"], "type": "categorical", "timing_justification": "Identifies observed, unseen, missing, and cold-start industry history without using candidate outcomes.", "missing_value_policy": "Explicit fallback category.", "cold_start_policy": "cold_start_global_fallback", "leakage_classification": "strict_prior_history", "available_during_candidate_scoring": True},
    ])
    return metadata


def _csv_text(rows: list[dict[str, Any]], columns: list[str]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_artifact_directory(directory: Path, expected_manifest: dict[str, Any]) -> tuple[bool, str]:
    """Validate every reusable artifact contract without trusting its manifest."""
    required_files = {"feature_manifest.json", *FEATURE_OUTPUT_FILES.values()}
    if not directory.is_dir():
        return False, "final path is not a directory"
    missing = sorted(name for name in required_files if not (directory / name).is_file())
    if missing:
        return False, f"missing required files: {', '.join(missing)}"
    try:
        manifest = json.loads((directory / "feature_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"manifest does not parse: {error}"
    for key in ("source_snapshot_id", "feature_version", "output_row_counts", "output_columns"):
        if manifest.get(key) != expected_manifest.get(key):
            return False, f"manifest {key} does not match the requested artifact"
    hashes = manifest.get("file_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(FEATURE_OUTPUT_FILES.values()):
        return False, "manifest does not contain hashes for exactly the required feature files"
    for split, filename in FEATURE_OUTPUT_FILES.items():
        path = directory / filename
        if sha256_bytes(path.read_bytes()) != hashes[filename]:
            return False, f"stored hash does not match {filename}"
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                row_count = sum(1 for _ in reader)
        except (OSError, csv.Error) as error:
            return False, f"could not validate {filename}: {error}"
        expected_columns = expected_manifest["output_columns"][split]
        expected_count = expected_manifest["output_row_counts"][split]
        if columns != expected_columns:
            return False, f"schema mismatch for {filename}"
        if row_count != expected_count:
            return False, f"row-count mismatch for {filename}"
    if TARGET_COLUMN in expected_manifest["output_columns"]["test_locked"]:
        return False, "locked-test schema contains the target"
    return True, "valid"


def _quarantine_path(directory: Path) -> Path:
    return directory.with_name(f"{directory.name}.quarantine-{uuid.uuid4().hex}")


def write_feature_artifacts(directory: Path, outputs: dict[str, tuple[list[dict[str, Any]], list[str]]], manifest: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Reuse only fully validated artifacts, otherwise rebuild through a sibling temporary directory."""
    if set(outputs) != set(FEATURE_OUTPUT_FILES.values()):
        raise FeatureEngineeringError("Feature artifact set must contain train, validation, and locked-test files.")
    rendered = {name: _csv_text(rows, columns).encode("utf-8") for name, (rows, columns) in outputs.items()}
    planned = {name: sha256_bytes(contents) for name, contents in rendered.items()}
    full_manifest = {**manifest, "file_hashes": planned}
    if directory.exists():
        valid, _reason = _validate_artifact_directory(directory, full_manifest)
        if valid:
            return read_feature_manifest(directory), True
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{directory.name}.tmp-", dir=directory.parent))
    try:
        for name, contents in rendered.items():
            (temporary / name).write_bytes(contents)
        (temporary / "feature_manifest.json").write_text(json.dumps(full_manifest, indent=2, sort_keys=True), encoding="utf-8")
        valid, reason = _validate_artifact_directory(temporary, full_manifest)
        if not valid:
            raise FeatureEngineeringError(f"Temporary feature artifacts failed validation: {reason}")
        if directory.exists():
            directory.replace(_quarantine_path(directory))
        temporary.replace(directory)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return full_manifest, False


def read_feature_manifest(directory: Path) -> dict[str, Any]:
    try:
        return json.loads((directory / "feature_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureEngineeringError(f"Could not read validated feature manifest: {directory}") from error


def missing_percentages(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, float]:
    return {column: round(100 * sum(row.get(column) in (None, "") for row in rows) / len(rows), 2) if rows else 100.0 for column in columns}


def numeric_ranges(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for column in columns:
        values = [row[column] for row in rows if row.get(column) is not None]
        result[column] = {"min": min(values) if values else None, "max": max(values) if values else None}
    return result
