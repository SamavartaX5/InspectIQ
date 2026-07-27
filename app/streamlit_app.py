"""InspectIQ's local human-review dashboard; it never loads candidate labels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dashboard import DashboardError, filter_candidates, load_dashboard_context, queue_csv
from src.explanations import global_feature_importance, local_perturbation_explanation, training_references


DISCLAIMER = "This retrospective prototype ranks a supplied candidate batch for human review. It does not automatically initiate enforcement."
SCORE_CAVEAT = "The score is an uncalibrated model output used for relative ranking. It is not a verified probability that a serious violation will be found."
NAVIGATION = ["Review Queue", "Candidate Detail", "Model Evidence", "Data & Limitations"]
CHART_LABELS = {
    "highest_priority": "Highest", "high_priority": "High", "elevated_priority": "Elevated",
    "standard_priority": "Standard", "nr_in_estab": "Establishment size",
    "industry_prior_inspection_count": "Prior industry inspections",
    "industry_prior_positive_count": "Prior industry positive findings",
    "industry_prior_positive_rate_smoothed": "Smoothed prior rate",
    "industry_history_status": "Industry history status",
}


def _streamlit():
    try:
        import streamlit as st
        return st
    except ImportError as exc:  # Keeps import/test validation possible before the user installs the optional UI dependency.
        raise RuntimeError("Streamlit is required to run the dashboard. Install the project's requirements first.") from exc


def _display_columns() -> dict[str, str]:
    return {
        "rank": "Rank", "activity_nr": "Activity ID", "open_date": "Inspection date", "raw_risk_score": "Raw risk score",
        "score_percentile": "Score percentile", "review_priority": "Review priority", "naics_group": "NAICS group",
        "insp_type": "Inspection type", "insp_scope": "Inspection scope", "owner_type": "Owner type",
        "safety_hlth": "Safety/health", "nr_in_estab": "Reported establishment size",
        "industry_prior_inspection_count": "Prior industry inspections",
        "industry_prior_positive_count": "Prior industry positive findings",
        "industry_prior_positive_rate_smoothed": "Smoothed prior positive rate",
        "industry_history_status": "Industry history status",
    }


def _short_chart_labels(values: pd.Series, prefix: str = "") -> pd.Series:
    """Keep native chart axes legible without changing the underlying data."""
    result = values.copy()
    result.index = [CHART_LABELS.get(str(value), f"{prefix}{value}") for value in result.index]
    return result


def _filters(st, ranked: pd.DataFrame) -> dict:
    if st.sidebar.button("Reset filters"):
        st.session_state.pop("dashboard_filters", None)
    state = st.session_state.setdefault("dashboard_filters", {})
    choices = {}
    for column, label in (("review_priority", "Review priority"), ("naics_group", "NAICS group"), ("insp_type", "Inspection type"), ("insp_scope", "Inspection scope"), ("owner_type", "Owner type"), ("safety_hlth", "Safety/health")):
        choices[column] = st.sidebar.multiselect(label, sorted(ranked[column].dropna().astype(str).unique().tolist()), default=state.get(column, []), key=f"filter_{column}")
    low, high = float(ranked.raw_risk_score.min()), float(ranked.raw_risk_score.max())
    choices["raw_risk_score"] = st.sidebar.slider("Raw score range", min_value=low, max_value=high, value=state.get("raw_risk_score", (low, high)), key="filter_raw_risk_score")
    st.session_state["dashboard_filters"] = choices
    return choices


def _queue_page(st, context, filtered: pd.DataFrame) -> None:
    st.subheader("Review Queue")
    cards = st.columns(3)
    cards[0].metric("Candidates in view", len(filtered))
    cards[1].metric("Highest-priority", int((filtered.review_priority == "highest_priority").sum()))
    cards[2].metric("Top-10% capacity", context.batch["top_counts"]["top_10_percent"])
    model_cards = st.columns(2)
    model_cards[0].metric("Selected model", "Random Forest")
    model_cards[1].metric("Score interpretation", "Uncalibrated")
    st.caption(SCORE_CAVEAT)
    counts = filtered.review_priority.value_counts().reindex(["highest_priority", "high_priority", "elevated_priority", "standard_priority"], fill_value=0)
    first, second = st.columns(2)
    first.caption("Review-priority counts")
    first.bar_chart(_short_chart_labels(counts.rename("candidates")), height=260)
    second.caption("Candidates by NAICS group")
    naics_counts = filtered.groupby("naics_group").size().sort_values(ascending=False).head(12).rename("candidates")
    second.bar_chart(_short_chart_labels(naics_counts, "NAICS "), height=260)
    st.caption("Candidates by inspection type")
    type_counts = filtered.groupby("insp_type").size().sort_values(ascending=False).head(12).rename("candidates")
    st.bar_chart(_short_chart_labels(type_counts, "Type "), height=240)
    st.caption("Raw risk-score distribution")
    st.bar_chart(filtered["raw_risk_score"].value_counts(bins=20, sort=False).rename("candidates"), height=240)
    table_columns = ["rank", "activity_nr", "open_date", "raw_risk_score", "score_percentile", "review_priority", "naics_group", "insp_type", "insp_scope", "owner_type", "safety_hlth", "nr_in_estab", "industry_prior_inspection_count", "industry_prior_positive_rate_smoothed", "industry_history_status"]
    st.dataframe(filtered.loc[:, table_columns].rename(columns=_display_columns()), use_container_width=True, hide_index=True, column_config={"Raw risk score": st.column_config.NumberColumn(format="%.4f"), "Score percentile": st.column_config.NumberColumn(format="%.1f")})
    st.download_button("Download filtered review queue", queue_csv(filtered), "inspectiq_filtered_review_queue.csv", "text/csv")
    st.download_button("Download complete ranked queue", queue_csv(context.ranked), "inspectiq_ranked_candidate_queue.csv", "text/csv")
    st.download_button("Download top-10% queue", queue_csv(context.top_10), "inspectiq_top_10_percent_queue.csv", "text/csv")


def _detail_page(st, context) -> None:
    st.subheader("Candidate Detail")
    selected_rank = st.selectbox("Candidate", context.ranked.rank.tolist(), format_func=lambda rank: f"Rank {rank} — activity {context.ranked.loc[context.ranked.rank == rank, 'activity_nr'].iloc[0]}")
    candidate = context.ranked.loc[context.ranked.rank == selected_rank].iloc[0]
    st.info(SCORE_CAVEAT)
    fields = ["rank", "raw_risk_score", "score_percentile", "review_priority", "open_date", "naics_group", "insp_type", "insp_scope", "owner_type", "safety_hlth", "nr_in_estab", "industry_prior_inspection_count", "industry_prior_positive_count", "industry_prior_positive_rate_smoothed", "industry_history_status"]
    st.dataframe(pd.DataFrame({"Field": [_display_columns().get(field, field) for field in fields], "Value": [candidate[field] for field in fields]}), hide_index=True, use_container_width=True)
    explanation = local_perturbation_explanation(context.model, candidate, training_references(context.training))
    st.markdown("#### Local score influences")
    st.caption(explanation.attrs["caveat"])
    explanation_chart = explanation.set_index("feature")["raw_score_difference"]
    st.bar_chart(_short_chart_labels(explanation_chart), height=360)
    st.dataframe(explanation.rename(columns={"feature_label": "Feature", "observed_value": "Observed value", "reference_value": "Training reference", "raw_score_difference": "Score difference", "direction": "Local influence"}).loc[:, ["Feature", "Observed value", "Training reference", "Score difference", "Local influence"]], hide_index=True, use_container_width=True)


def _evidence_page(st, context) -> None:
    st.subheader("Model Evidence")
    st.caption("These are recorded 2022 chronological validation results. They are not 2023 candidate performance metrics.")
    selected = next(item for item in context.comparison["experiments"] if item["experiment_id"] == context.comparison["selected_candidate"])
    baseline = context.comparison["baseline"]
    metric = selected["metrics"]
    columns = st.columns(4)
    columns[0].metric("Baseline positives in top 60", baseline["positives_captured_top_10"])
    columns[1].metric("Selected positives in top 60", metric["ranking_at"]["10"]["selected_positives"])
    columns[2].metric("Selected Recall@10", f"{metric['ranking_at']['10']['recall']:.2%}")
    columns[3].metric("Selected Precision@10", f"{metric['ranking_at']['10']['precision']:.2%}")
    st.json({"validation_lift_at_10": metric["ranking_at"]["10"]["lift"], "validation_pr_auc": metric["pr_auc"], "validation_roc_auc": metric["roc_auc"], "validation_brier_score": metric["brier_score"]})
    st.markdown("#### Calibration conclusion")
    st.write(context.calibration["selection_rationale"])
    st.write(f"Selected calibration method: **{context.calibration['selected_calibration_method']}**. The final candidate score remains uncalibrated.")
    st.markdown("#### Local MLflow tracking")
    st.write(f"Status: {context.mlflow['status']}; Day 3 runs: {context.mlflow['logged_day3_run_count']}; Day 4 runs: {context.mlflow['logged_day4_run_count']}.")
    st.markdown("#### Global fitted-model importance")
    importance = global_feature_importance(context.model)
    st.caption("Importance reflects how much the fitted Random Forest used a feature across its trees; it does not show that a feature causes violations.")
    importance_chart = importance.set_index("feature")["importance"]
    st.bar_chart(_short_chart_labels(importance_chart), height=360)


def _limitations_page(st, context) -> None:
    st.subheader("Data & Limitations")
    st.json({"source_snapshot_id": context.batch["source_snapshot_id"], "feature_version": context.batch["feature_version"], "candidate_date_range": context.batch["open_date_range"], "candidate_row_count": context.batch["input_row_count"], "selected_experiment": context.batch["selected_day3_experiment"], "selected_calibration_method": context.batch["selected_day4_method"], "score_terminology": "raw_risk_score (uncalibrated model output)", "ranking_rule": context.prediction_manifest["ranking_rule"]})
    for limitation in [
        "The 2023 candidate batch has no labels available to this workflow.", "No out-of-time test performance metric has been calculated.",
        "Scores are uncalibrated model outputs and are not verified probabilities.", "Historical OSHA inspection data is selection-biased and covers only represented historical inspections.",
        "The model ranks only the supplied candidate batch.", "Explanations describe model behaviour, not causality.",
        "Human review is required; the system does not automatically initiate enforcement.",
    ]:
        st.write("• " + limitation)


def main() -> None:
    st = _streamlit()
    st.set_page_config(page_title="InspectIQ", page_icon="🔎", layout="wide")
    st.title("InspectIQ — Workplace Safety Inspection Risk Triage")
    st.caption(DISCLAIMER)
    try:
        context = load_dashboard_context()
    except DashboardError as exc:
        st.error(f"Dashboard artifact validation failed: {exc}")
        st.stop()
        return
    st.sidebar.header("Review queue controls")
    page = st.sidebar.radio("Navigate", NAVIGATION)
    filtered = filter_candidates(context.ranked, _filters(st, context.ranked))
    if page == "Review Queue":
        _queue_page(st, context, filtered)
    elif page == "Candidate Detail":
        _detail_page(st, context)
    elif page == "Model Evidence":
        _evidence_page(st, context)
    else:
        _limitations_page(st, context)


if __name__ == "__main__":
    main()
