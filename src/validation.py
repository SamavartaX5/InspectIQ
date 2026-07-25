"""Validation contracts for the cache-only Day 1 data foundation."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


class ValidationError(RuntimeError):
    """Input or output violates the documented Day 1 contract."""


def load_schema(path: Path) -> dict[str, Any]:
    """The schema file is JSON-formatted YAML, so no parser dependency is needed."""
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"Could not load schema {path}: {error}") from error
    if not isinstance(schema.get("inspection_fields"), list):
        raise ValidationError("Schema does not define inspection_fields.")
    return schema


def blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def parse_date(value: Any) -> datetime | None:
    if blank(value):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_inspections(
    rows: list[dict[str, Any]], schema: dict[str, Any], *, state: str, start_date: str, end_date: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fields = schema["inspection_fields"]
    required = [item["name"] for item in fields if item.get("required")]
    non_null = [item["name"] for item in fields if item.get("non_null")]
    allowed = {item["name"]: set(item.get("allowed_values", [])) for item in fields if item.get("allowed_values")}
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    unknown_values: Counter[str] = Counter()
    invalid_dates = 0
    duplicate_count = 0
    for row in rows:
        reasons = [f"missing_required_column:{name}" for name in required if name not in row]
        reasons.extend(f"missing_required_value:{name}" for name in non_null if name in row and blank(row[name]))
        activity = None if blank(row.get("activity_nr")) else str(row.get("activity_nr")).strip()
        if activity is None:
            reasons.append("null_activity_nr")
        elif activity in seen:
            duplicate_count += 1
            reasons.append("duplicate_activity_nr")
        opened = parse_date(row.get("open_date"))
        if opened is None:
            invalid_dates += 1
            reasons.append("invalid_open_date")
        elif not (start <= opened.replace(tzinfo=None) <= end):
            reasons.append("open_date_outside_requested_range")
        if str(row.get("site_state", "")).upper() != state.upper():
            reasons.append("unexpected_site_state")
        for name, values in allowed.items():
            value = str(row.get(name, "")).strip()
            if value and value not in values:
                unknown_values[f"{name}:{value}"] += 1
        if reasons:
            rejected.append({**row, "rejection_reason": ";".join(sorted(set(reasons)))})
            continue
        seen.add(activity)
        accepted.append(row)
    metrics = {
        "duplicate_inspection_count": duplicate_count,
        "invalid_date_count": invalid_dates,
        "unknown_categorical_values": dict(unknown_values),
        "rejected_reason_counts": dict(Counter(row["rejection_reason"] for row in rejected)),
    }
    return accepted, rejected, metrics


def missing_percentages(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, float]:
    if not rows:
        return {column: 100.0 for column in columns}
    return {
        column: round(100 * sum(blank(row.get(column)) for row in rows) / len(rows), 2)
        for column in columns
    }


def validate_output_columns(columns: list[str], schema: dict[str, Any]) -> None:
    required = [item["name"] for item in schema["inspection_fields"] if item.get("feature")]
    missing = [name for name in required + ["serious_violation_found"] if name not in columns]
    prohibited = sorted(set(columns) & set(schema["prohibited_leakage_columns"]))
    if missing:
        raise ValidationError(f"Labelled output is missing required columns: {', '.join(missing)}")
    if prohibited:
        raise ValidationError(f"Labelled output contains prohibited leakage columns: {', '.join(prohibited)}")
