"""Small dependency-free ranking metrics used for validation only."""

from __future__ import annotations

import math
from typing import Any


def selection_count(row_count: int, proportion: float) -> int:
    if row_count <= 0:
        raise ValueError("Cannot select from zero validation rows.")
    return max(1, math.ceil(row_count * proportion))


def ranking_at_fraction(rows: list[dict[str, Any]], proportion: float) -> dict[str, Any]:
    selected = selection_count(len(rows), proportion)
    top = rows[:selected]
    positives = sum(int(row["actual_label"]) for row in rows)
    selected_positives = sum(int(row["actual_label"]) for row in top)
    precision = selected_positives / selected
    recall = selected_positives / positives if positives else None
    base_rate = positives / len(rows)
    return {
        "fraction": proportion, "selected": selected, "selected_positives": selected_positives,
        "precision": precision, "recall": recall, "lift": precision / base_rate if base_rate else None,
    }


def average_precision(rows: list[dict[str, Any]]) -> float | None:
    positives = sum(int(row["actual_label"]) for row in rows)
    if not positives:
        return None
    seen_positives = 0
    total = 0.0
    for index, row in enumerate(rows, start=1):
        if int(row["actual_label"]):
            seen_positives += 1
            total += seen_positives / index
    return total / positives


def roc_auc(rows: list[dict[str, Any]]) -> float | None:
    positive_scores = [float(row["baseline_score"]) for row in rows if int(row["actual_label"]) == 1]
    negative_scores = [float(row["baseline_score"]) for row in rows if int(row["actual_label"]) == 0]
    if not positive_scores or not negative_scores:
        return None
    wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive in positive_scores for negative in negative_scores)
    return wins / (len(positive_scores) * len(negative_scores))


def validation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "positive_count": sum(int(row["actual_label"]) for row in rows),
        "negative_count": sum(1 - int(row["actual_label"]) for row in rows),
        "ranking_at": {str(int(proportion * 100)): ranking_at_fraction(rows, proportion) for proportion in (0.05, 0.10, 0.20)},
        "pr_auc": average_precision(rows),
        "average_precision": average_precision(rows),
        "roc_auc": roc_auc(rows),
    }
