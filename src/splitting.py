"""Leakage-free chronological split helpers for the Day 1 baseline."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data_foundation import sha256_file, write_json


class SplitError(RuntimeError):
    """The labelled snapshot cannot support a defensible chronological split."""


def parse_open_date(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise SplitError(f"Invalid open_date in labelled data: {value!r}") from error


def activity_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (parse_open_date(row["open_date"]), activity_sort_key(row["activity_nr"])))


def _period_summary(rows: list[dict[str, Any]], years: list[int]) -> dict[str, Any]:
    dates = [parse_open_date(row["open_date"]).date().isoformat() for row in rows]
    return {
        "years": years,
        "row_count": len(rows),
        "open_date_range": {"min": min(dates), "max": max(dates)},
    }


def create_chronological_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a three-period split, selecting the documented preferred split when possible."""
    if not rows:
        raise SplitError("No labelled records are available for chronological splitting.")
    ids = [str(row.get("activity_nr", "")) for row in rows]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise SplitError("Labelled records must have unique, non-empty activity_nr values.")

    by_year: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_year.setdefault(parse_open_date(row["open_date"]).year, []).append(row)
    years = sorted(by_year)
    if len(years) < 3:
        raise SplitError("At least three distinct labelled calendar years are required.")

    preferred = {"train": [2020, 2021, 2022], "validation": [2023], "test": [2024]}
    if all(year in by_year for period in preferred.values() for year in period):
        selected = preferred
        strategy = "preferred_2020_2022_train_2023_validation_2024_test"
        rationale = "The preferred requested calendar-year split is fully represented in labelled data."
    else:
        selected = {"train": years[:-2], "validation": [years[-2]], "test": [years[-1]]}
        strategy = "fallback_earliest_train_penultimate_validation_latest_test"
        rationale = (
            "The preferred 2020-2024 split is not fully labelled; used the earliest available years for training, "
            "the penultimate labelled year for validation, and the latest labelled year as the locked test period."
        )

    split_rows = {
        period: sort_rows([row for year in period_years for row in by_year[year]])
        for period, period_years in selected.items()
    }
    if any(not values for values in split_rows.values()):
        raise SplitError("Train, validation, and locked test periods must all be non-empty.")
    train_ids, validation_ids, test_ids = (set(str(row["activity_nr"]) for row in split_rows[name]) for name in ("train", "validation", "test"))
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise SplitError("activity_nr values overlap between chronological split periods.")

    train_max = max(parse_open_date(row["open_date"]) for row in split_rows["train"])
    validation_min = min(parse_open_date(row["open_date"]) for row in split_rows["validation"])
    validation_max = max(parse_open_date(row["open_date"]) for row in split_rows["validation"])
    test_min = min(parse_open_date(row["open_date"]) for row in split_rows["test"])
    if not train_max < validation_min < test_min or not validation_max < test_min:
        raise SplitError("Chronological split date boundaries overlap or are not strictly ordered.")

    return {
        "strategy": strategy,
        "rationale": rationale,
        "preferred_split": preferred,
        "available_labelled_years": years,
        "selected_years": selected,
        "rows": split_rows,
        "periods": {name: _period_summary(split_rows[name], selected[name]) for name in split_rows},
        "strictly_ordered": True,
        "id_overlap_count": 0,
    }


def write_split_artifacts(directory: Path, split: dict[str, Any]) -> dict[str, Any]:
    """Persist deterministic split CSVs; the locked test file intentionally has no label column."""
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for period, filename in (("train", "train.csv"), ("validation", "validation.csv"), ("test", "test_locked.csv")):
        rows = split["rows"][period]
        columns = list(rows[0].keys())
        if period == "test":
            columns = [column for column in columns if column != "serious_violation_found"]
        path = directory / filename
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
        files[filename] = sha256_file(path)
    manifest = {
        "strategy": split["strategy"], "rationale": split["rationale"],
        "preferred_split": split["preferred_split"], "available_labelled_years": split["available_labelled_years"],
        "selected_years": split["selected_years"], "periods": split["periods"],
        "strictly_ordered": split["strictly_ordered"], "id_overlap_count": split["id_overlap_count"],
        "file_hashes": files,
    }
    write_json(directory / "split_manifest.json", manifest)
    return manifest
