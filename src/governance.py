"""Deterministic human-review and future-label templates; no outcomes are created."""
from __future__ import annotations
import pandas as pd

REVIEW_COLUMNS = ["rank","activity_nr","open_date","raw_risk_score","review_priority","top_10_percent_flag","prediction_run_id","human_review_status","reviewer_decision","reviewer_reason","reviewer_notes","reviewed_at","override_model_priority","escalation_required"]
OUTCOME_COLUMNS = ["activity_nr","outcome_observed_date","serious_violation_found","label_source","label_completeness_status"]
ALLOWED_DECISIONS = ["pending","review_later","request_additional_information","escalate_for_human_assessment","no_further_review"]

def review_template(ranked: pd.DataFrame, prediction_run_id: str) -> pd.DataFrame:
    fixed = ranked[["rank","activity_nr","open_date","raw_risk_score","review_priority","top_10_percent_flag"]].copy()
    fixed["prediction_run_id"] = prediction_run_id
    for name in REVIEW_COLUMNS[7:]: fixed[name] = ""
    return fixed.loc[:, REVIEW_COLUMNS]

def future_outcome_template() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTCOME_COLUMNS)
