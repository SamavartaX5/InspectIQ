"""Transparent NAICS historical-rate ranking baseline trained only on training rows."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from src.splitting import activity_sort_key


DEFAULT_ALPHA = 5.0
DEFAULT_MIN_GROUP_SIZE = 20


def naics_group(value: Any, digits: int) -> str | None:
    raw = str(value or "").strip()
    raw = raw.split(".", 1)[0]
    numbers = "".join(re.findall(r"\d", raw))
    return numbers[:digits] if len(numbers) >= digits else None


def _candidate_summary(rows: list[dict[str, Any]], digits: int, minimum_group_size: int) -> dict[str, Any]:
    groups = [naics_group(row.get("naics_code"), digits) for row in rows]
    counts = Counter(group for group in groups if group is not None)
    adequate_rows = sum(1 for group in groups if group is not None and counts[group] >= minimum_group_size)
    return {
        "digits": digits, "non_missing_rows": sum(group is not None for group in groups),
        "group_count": len(counts), "adequate_group_count": sum(count >= minimum_group_size for count in counts.values()),
        "adequate_row_coverage": adequate_rows / len(rows) if rows else 0.0,
        "group_sizes": dict(sorted(counts.items())),
    }


def choose_naics_grouping(training_rows: list[dict[str, Any]], minimum_group_size: int = DEFAULT_MIN_GROUP_SIZE) -> tuple[int, dict[str, Any]]:
    candidates = {digits: _candidate_summary(training_rows, digits, minimum_group_size) for digits in (3, 2)}
    selected = 3 if candidates[3]["adequate_row_coverage"] >= 0.80 else 2
    return selected, {"candidates": candidates, "selected_digits": selected, "selection_rule": "Use 3-digit NAICS when at least 80% of training rows are in groups meeting the minimum size; otherwise use 2-digit NAICS."}


def build_industry_rates(training_rows: list[dict[str, Any]], digits: int, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    if not training_rows:
        raise ValueError("Training rows are required to build an industry-rate baseline.")
    labels = [int(row["serious_violation_found"]) for row in training_rows]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Training labels must be binary.")
    global_rate = sum(labels) / len(labels)
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, label in zip(training_rows, labels):
        group = naics_group(row.get("naics_code"), digits)
        if group is not None:
            grouped[group].append(label)
    rates = {
        group: {"row_count": len(values), "positive_count": sum(values), "smoothed_rate": (sum(values) + alpha * global_rate) / (len(values) + alpha)}
        for group, values in sorted(grouped.items())
    }
    return {"global_positive_rate": global_rate, "alpha": alpha, "digits": digits, "groups": rates}


def score_validation_rows(
    validation_rows: list[dict[str, Any]], rates: dict[str, Any], minimum_group_size: int = DEFAULT_MIN_GROUP_SIZE
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in validation_rows:
        group = naics_group(row.get("naics_code"), int(rates["digits"]))
        if group is None:
            score, source, display_group = rates["global_positive_rate"], "missing_industry_fallback", None
        elif group not in rates["groups"]:
            score, source, display_group = rates["global_positive_rate"], "unseen_group_fallback", group
        elif rates["groups"][group]["row_count"] < minimum_group_size:
            score, source, display_group = rates["global_positive_rate"], "sparse_group_fallback", group
        else:
            score, source, display_group = rates["groups"][group]["smoothed_rate"], "industry_rate", group
        scored.append({
            "activity_nr": str(row["activity_nr"]), "open_date": str(row["open_date"]), "industry_group": display_group,
            "baseline_score": score, "actual_label": int(row["serious_violation_found"]), "score_source": source,
        })
    scored.sort(key=lambda row: (-float(row["baseline_score"]), activity_sort_key(row["activity_nr"])))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    return scored
