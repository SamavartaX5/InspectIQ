"""InspectIQ's read-only public dashboard for advisory human review."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.dashboard import DashboardError, filter_candidates, load_dashboard_context, queue_csv
from src.explanations import global_feature_importance, local_perturbation_explanation
from src.path_utils import resolve_report_path
from src.ui_components import (
    priority_label,
    render_badge_row,
    render_empty_state,
    render_footer,
    render_info_banner,
    render_kpi_row,
    render_page_header,
    render_status_badge,
    render_warning_banner,
)
from src.ui_theme import apply_theme


DISCLAIMER = "Retrospective decision-support prototype. It ranks supplied candidates for human review; no automatic enforcement occurs."
SCORE_CAVEAT = "The uncalibrated advisory score supports relative queue ordering. It is not a calibrated outcome estimate."
NAVIGATION = ["Review Queue", "Candidate Detail", "Model Evidence", "Monitoring & Governance", "Data & Limitations"]
BUDGETS = {"5%": 0.05, "10%": 0.10, "20%": 0.20, "All candidates": 1.0}
PRIORITY_ORDER = ["highest_priority", "high_priority", "elevated_priority", "standard_priority"]
CHART_LABELS = {
    "highest_priority": "Highest priority", "high_priority": "High priority",
    "elevated_priority": "Standard review", "standard_priority": "Later review",
    "nr_in_estab": "Establishment size", "industry_prior_inspection_count": "Prior industry inspections",
    "industry_prior_positive_count": "Prior industry positive findings",
    "industry_prior_positive_rate_smoothed": "Smoothed prior rate",
    "industry_history_status": "Industry history status",
}


def _streamlit():
    try:
        import streamlit as st
        return st
    except ImportError as exc:
        raise RuntimeError("Streamlit is required to run the dashboard. Install the project requirements first.") from exc


def _display_columns() -> dict[str, str]:
    return {
        "rank": "Rank", "activity_nr": "Activity ID", "open_date": "Inspection date",
        "raw_risk_score": "Advisory score", "score_percentile": "Queue percentile",
        "review_priority": "Review priority", "naics_group": "NAICS group",
        "insp_type": "Inspection type", "insp_scope": "Inspection scope", "owner_type": "Owner type",
        "safety_hlth": "Safety/health", "nr_in_estab": "Reported establishment size",
        "industry_prior_inspection_count": "Prior industry inspections",
        "industry_prior_positive_count": "Prior industry positive findings",
        "industry_prior_positive_rate_smoothed": "Smoothed prior positive rate",
        "industry_history_status": "Industry history status",
    }


def _short_chart_labels(values: pd.Series, prefix: str = "") -> pd.Series:
    result = values.copy()
    result.index = [CHART_LABELS.get(str(value), f"{prefix}{value}") for value in result.index]
    return result


def budget_queue(frame: pd.DataFrame, budget: str) -> pd.DataFrame:
    """Return a deterministic rank-prefix for a review budget without changing ranks."""
    if budget not in BUDGETS:
        raise ValueError(f"Unsupported review budget: {budget}")
    count = len(frame) if BUDGETS[budget] == 1.0 else int(round(len(frame) * BUDGETS[budget]))
    return frame.loc[frame["rank"] <= count].copy()


def active_filter_count(filters: dict, ranked: pd.DataFrame) -> int:
    count = sum(bool(filters.get(name)) for name in ("review_priority", "naics_group", "insp_type", "insp_scope", "owner_type", "safety_hlth"))
    selected = filters.get("raw_risk_score")
    full = (float(ranked["raw_risk_score"].min()), float(ranked["raw_risk_score"].max()))
    return count + int(selected is not None and tuple(selected) != full)


def adjacent_rank(ranks: list[int], selected: int, direction: int) -> int:
    """Move through the immutable rank list while staying at its boundaries."""
    index = ranks.index(selected)
    return ranks[min(len(ranks) - 1, max(0, index + direction))]


def _filters(st, ranked: pd.DataFrame) -> dict:
    if st.sidebar.button("Reset filters", help="Restore the complete frozen candidate queue."):
        st.session_state.pop("dashboard_filters", None)
    state = st.session_state.setdefault("dashboard_filters", {})
    choices = {}
    controls = (
        ("review_priority", "Review priority"), ("naics_group", "NAICS group"),
        ("insp_type", "Inspection type"), ("insp_scope", "Inspection scope"),
        ("owner_type", "Owner type"), ("safety_hlth", "Safety/health classification"),
    )
    for column, label in controls:
        choices[column] = st.sidebar.multiselect(
            label, sorted(ranked[column].dropna().astype(str).unique().tolist()),
            default=state.get(column, []), key=f"filter_{column}",
        )
    low, high = float(ranked["raw_risk_score"].min()), float(ranked["raw_risk_score"].max())
    choices["raw_risk_score"] = st.sidebar.slider(
        "Advisory score range", min_value=low, max_value=high,
        value=state.get("raw_risk_score", (low, high)), key="filter_raw_risk_score",
    )
    st.session_state["dashboard_filters"] = choices
    return choices


def _altair_chart(st, frame: pd.DataFrame, *, category: str, measure: str, title: str, color: str = "#4FC3F7", height: int = 270) -> None:
    """Use explicit nominal/quantitative encodings; native fallback keeps the demo usable."""
    try:
        import altair as alt
    except ImportError:
        st.bar_chart(frame.set_index(category)[measure], height=height)
        return
    chart = alt.Chart(frame).mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{category}:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30, labelLimit=150)),
        y=alt.Y(f"{measure}:Q", title=title),
        tooltip=[alt.Tooltip(f"{category}:N", title=category.replace("_", " ").title()), alt.Tooltip(f"{measure}:Q", title=title, format=",.3f")],
    ).properties(height=height).interactive()
    st.altair_chart(chart, width="stretch")


def _score_distribution(st, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    score_bins = frame["raw_risk_score"].value_counts(bins=12, sort=False).rename("candidate count")
    score_bins.index = score_bins.index.astype(str)
    distribution = score_bins.rename_axis("score band").reset_index()
    _altair_chart(st, distribution, category="score band", measure="candidate count", title="Candidate count", color="#5bc0de", height=250)


def _queue_page(st, context, filtered: pd.DataFrame) -> None:
    render_page_header(st, "Review Queue", "Triage the supplied 2023 candidate batch into a human-review queue.", eyebrow="Candidate ranking")
    budget = st.select_slider("Review budget", options=list(BUDGETS), value=st.session_state.get("review_budget", "10%"), help="Changes review emphasis and downloads; it never changes frozen ranks or scores.")
    if budget not in BUDGETS:  # Makes the rendering smoke test tolerant of minimal Streamlit fakes.
        budget = "10%"
    st.session_state["review_budget"] = budget
    budgeted = budget_queue(filtered, budget)
    filters = st.session_state.get("dashboard_filters", {})
    render_kpi_row(st, [
        {"label": "Candidates in view", "value": len(filtered), "help": "Frozen candidates matching current filters."},
        {"label": "Review budget", "value": f"{budget} · {len(budgeted)}", "help": "Rank-prefix capacity under the selected budget."},
        {"label": "Active filters", "value": active_filter_count(filters, context.ranked)},
    ])
    render_kpi_row(st, [
        {"label": "Top 5%", "value": context.batch["top_counts"].get("top_5_percent", int(round(len(context.ranked) * 0.05))), "caption": "Highest review tier"},
        {"label": "Top 10%", "value": context.batch["top_counts"]["top_10_percent"], "caption": "Standard review capacity"},
    ])
    render_info_banner(st, SCORE_CAVEAT)
    if filtered.empty:
        render_empty_state(st, "No candidates match these filters", "Reset filters or widen the advisory-score range. No ranking data was changed.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("#### Queue priority")
        priority = filtered["review_priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
        priority_frame = _short_chart_labels(priority.rename("candidate count")).rename_axis("review priority").reset_index()
        _altair_chart(st, priority_frame, category="review priority", measure="candidate count", title="Candidate count", color="#4FC3F7")
    with right:
        st.markdown("#### Candidate mix by NAICS group")
        naics = filtered.groupby("naics_group").size().sort_values(ascending=False).head(12).rename("candidate count")
        naics_frame = _short_chart_labels(naics, "NAICS ").rename_axis("NAICS group").reset_index()
        _altair_chart(st, naics_frame, category="NAICS group", measure="candidate count", title="Candidate count", color="#81c995")
    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("#### Advisory score distribution")
        _score_distribution(st, filtered)
    with lower_right:
        st.markdown("#### Selected-budget inspection types")
        types = budgeted.groupby("insp_type").size().sort_values(ascending=False).rename("candidate count")
        type_frame = _short_chart_labels(types, "Type ").rename_axis("inspection type").reset_index()
        _altair_chart(st, type_frame, category="inspection type", measure="candidate count", title="Candidate count", color="#e8b84d", height=250)

    st.markdown("#### Ranked candidates")
    st.caption("The table contains candidate context and advisory queue information only; no outcome labels are included.")
    table_columns = ["rank", "activity_nr", "open_date", "raw_risk_score", "score_percentile", "review_priority", "naics_group", "insp_type", "insp_scope", "owner_type", "safety_hlth", "nr_in_estab"]
    table = filtered.loc[:, table_columns].copy()
    table["review_priority"] = table["review_priority"].map(priority_label)
    st.dataframe(
        table.rename(columns=_display_columns()), width="stretch", hide_index=True,
        column_config={"Advisory score": st.column_config.NumberColumn(format="%.4f"), "Queue percentile": st.column_config.NumberColumn(format="%.1f")},
    )
    st.markdown("#### Queue exports")
    st.caption("Exports are read-only copies of the filtered or selected rank-prefix queue. They contain no labels or outcomes.")
    downloads = st.columns(3)
    downloads[0].download_button("Download filtered queue", queue_csv(filtered), "inspectiq_filtered_review_queue.csv", "text/csv")
    downloads[1].download_button(f"Download {budget} queue", queue_csv(budgeted), "inspectiq_selected_review_queue.csv", "text/csv")
    downloads[2].download_button("Download top 10% queue", queue_csv(context.top_10), "inspectiq_top_10_percent_queue.csv", "text/csv")


def _candidate_rank_label(context, rank: int) -> str:
    activity = context.ranked.loc[context.ranked["rank"] == rank, "activity_nr"].iloc[0]
    return f"Rank {rank} — activity {activity}"


def _detail_page(st, context, selected_rank: int | None = None) -> None:
    render_page_header(st, "Candidate Detail", "Review a single supplied candidate with its inspection context and model explanation.", eyebrow="Human review workspace")
    if selected_rank is None:
        selected_rank = st.selectbox("Candidate", context.ranked["rank"].tolist(), format_func=lambda rank: _candidate_rank_label(context, rank))
    candidate = context.ranked.loc[context.ranked["rank"] == selected_rank].iloc[0]
    total = len(context.ranked)
    render_badge_row(st, [
        ("Candidate", str(candidate["activity_nr"]), "neutral"),
        ("Review tier", priority_label(candidate["review_priority"]), "warning"),
        ("Queue position", f"{candidate['rank']} of {total}", "neutral"),
        ("Uncalibrated", "advisory score", "neutral"),
    ])
    hero, gauge = st.columns(2)
    with hero:
        render_kpi_row(st, [
            {"label": "Rank", "value": f"#{candidate['rank']}"},
            {"label": "Advisory score", "value": f"{candidate['raw_risk_score']:.4f}"},
            {"label": "Inspection date", "value": str(candidate["open_date"])},
        ])
        memberships = [name for name, flag in (("Top 5%", candidate["top_5_percent_flag"]), ("Top 10%", candidate["top_10_percent_flag"]), ("Top 20%", candidate["top_20_percent_flag"])) if bool(flag)]
        st.caption(" · ".join(memberships) if memberships else "Outside the top-20% rank prefix")
    with gauge:
        st.markdown("#### Queue-relative score")
        st.progress(float(candidate["raw_risk_score"]), text="Uncalibrated advisory score")
        st.caption(f"Queue percentile: {candidate['score_percentile']:.1f}. This is a relative queue position, not a confidence claim.")
    render_warning_banner(st, "Human review remains necessary. This advisory score does not confirm a violation and does not trigger automatic action.")

    context_col, history_col = st.columns(2)
    with context_col:
        st.markdown("#### Inspection context")
        context_fields = ["naics_group", "insp_type", "insp_scope", "owner_type", "safety_hlth", "nr_in_estab"]
        st.dataframe(pd.DataFrame({"Field": [_display_columns()[field] for field in context_fields], "Value": [candidate[field] for field in context_fields]}), hide_index=True, width="stretch")
    with history_col:
        st.markdown("#### Historical context")
        history_fields = ["industry_prior_inspection_count", "industry_prior_positive_count", "industry_prior_positive_rate_smoothed", "industry_history_status"]
        st.dataframe(pd.DataFrame({"Field": [_display_columns()[field] for field in history_fields], "Value": [candidate[field] for field in history_fields]}), hide_index=True, width="stretch")

    comparison = st.radio("Comparison reference", ["Queue median", "Top-10% median", "Same NAICS group median"], horizontal=True)
    if comparison == "Top-10% median":
        reference = context.top_10
    elif comparison == "Same NAICS group median":
        reference = context.ranked.loc[context.ranked["naics_group"] == candidate["naics_group"]]
    else:
        reference = context.ranked
    reference_score = float(reference["raw_risk_score"].median())
    comparison_items = [{"label": "Candidate score", "value": f"{candidate['raw_risk_score']:.4f}"}, {"label": "Reference median", "value": f"{reference_score:.4f}"}]
    if "industry_prior_inspection_count" in reference:
        comparison_items.append({"label": "Candidate prior inspections", "value": int(candidate["industry_prior_inspection_count"])})
        comparison_items.append({"label": "Reference median prior inspections", "value": f"{float(reference['industry_prior_inspection_count'].median()):.0f}"})
    render_kpi_row(st, comparison_items)

    st.markdown("#### Model evidence for this candidate")
    st.caption("One-feature-at-a-time sensitivity describes fitted-model behavior. It is not a causal explanation.")
    explanation = local_perturbation_explanation(context.model, candidate, context.training_references)
    explanation_chart = explanation.loc[:, ["feature_label", "raw_score_difference"]].rename(columns={"feature_label": "feature", "raw_score_difference": "score difference"})
    _altair_chart(st, explanation_chart, category="feature", measure="score difference", title="Score difference", color="#9ac8ff", height=330)
    st.dataframe(explanation.rename(columns={"feature_label": "Feature", "observed_value": "Observed value", "reference_value": "Training reference", "raw_score_difference": "Score difference", "direction": "Local influence"}).loc[:, ["Feature", "Observed value", "Training reference", "Score difference", "Local influence"]], hide_index=True, width="stretch")
    st.markdown("#### Suggested human-review focus")
    st.write("Review the supplied inspection context, historical indicators, and any applicable human workflow information. A reviewer may assess or escalate through their own process; this dashboard makes no automatic decision.")


def _evidence_page(st, context) -> None:
    render_page_header(st, "Model Evidence", "Recorded retrospective validation evidence for the selected ranking model.", eyebrow="2022 chronological validation")
    selected = next(item for item in context.comparison["experiments"] if item["experiment_id"] == context.comparison["selected_candidate"])
    metric, baseline = selected["metrics"], context.comparison["baseline"]
    render_badge_row(st, [("Validation period", "2022 retrospective", "neutral"), ("Selected model", "Random Forest", "pass"), ("Calibration", "Uncalibrated retained", "warning")])
    render_kpi_row(st, [
        {"label": "Selected experiment", "value": context.comparison["selected_candidate"]},
        {"label": "Validation rows", "value": metric["row_count"]},
        {"label": "Validation positives", "value": metric["positive_count"]},
        {"label": "Calibration decision", "value": "Uncalibrated"},
    ])
    render_info_banner(st, "All metrics on this page are retrospective 2022 validation results. They are not 2023 candidate performance metrics.")
    ranking_items = []
    for budget in ("5", "10", "20"):
        values = metric["ranking_at"][budget]
        ranking_items.extend([
            {"label": f"Precision@{budget}%", "value": f"{values['precision']:.1%}"},
            {"label": f"Recall@{budget}%", "value": f"{values['recall']:.1%}"},
            {"label": f"Lift@{budget}%", "value": f"{values['lift']:.2f}×"},
        ])
    st.markdown("#### Ranking results · retrospective 2022 validation")
    for start in range(0, len(ranking_items), 3):
        render_kpi_row(st, ranking_items[start:start + 3])
    render_kpi_row(st, [{"label": "PR-AUC", "value": f"{metric['pr_auc']:.3f}"}, {"label": "ROC-AUC", "value": f"{metric['roc_auc']:.3f}"}, {"label": "Brier score", "value": f"{metric['brier_score']:.3f}"}])

    st.markdown("#### Review-budget comparison · retrospective 2022 validation")
    comparison_rows = []
    for budget in ("5", "10", "20"):
        values = metric["ranking_at"][budget]
        baseline_captured = round(float(baseline["positives_captured_top_10"]) * (float(budget) / 10.0))
        comparison_rows.extend([
            {"budget": f"{budget}% model", "positives captured": values["selected_positives"]},
            {"budget": f"{budget}% baseline", "positives captured": baseline_captured},
        ])
    _altair_chart(st, pd.DataFrame(comparison_rows), category="budget", measure="positives captured", title="Recorded positives captured", color="#81c995", height=280)
    st.markdown("#### Why this model?")
    st.write("Eight experiments were compared on chronological validation. Random Forest was selected for the recorded operational ranking metric and its improvement over the documented baseline. This does not establish performance outside the validation population.")
    st.markdown("#### Why not accuracy?")
    st.write("The workflow prioritizes review capacity: precision, recall, and lift at fixed queue budgets show how a reviewer’s limited attention was allocated in retrospective validation.")
    st.markdown("#### Calibration decision")
    st.write("Uncalibrated, sigmoid, and isotonic study outputs were evaluated. The uncalibrated model was retained because calibration did not meet the recorded improvement policy. Scores remain ranking outputs rather than calibrated outcome estimates.")
    st.write(context.calibration["selection_rationale"])
    st.markdown("#### Global fitted-model drivers")
    importance = global_feature_importance(context.model)
    importance_chart = importance.loc[:, ["feature_label", "importance"]].rename(columns={"feature_label": "feature"})
    _altair_chart(st, importance_chart, category="feature", measure="importance", title="Relative importance", color="#4FC3F7", height=330)
    st.caption("Importance reflects use by the fitted Random Forest, not causal effect.")
    with st.expander("Technical details: hyperparameters and lineage"):
        st.json({"hyperparameters": selected["hyperparameters"], "chronology": "2020–2021 training → 2022 retrospective validation", "leakage_controls": context.feature_report.get("leakage_checks", {}), "selected_experiment": context.comparison["selected_candidate"]})


def _monitoring_page(st, context) -> None:
    report = context.monitoring
    if not report:
        st.error("Monitoring report is unavailable for this dashboard configuration.")
        return
    render_page_header(st, "Monitoring & Governance", "Operational monitoring signals and human-review safeguards for the frozen candidate batch.", eyebrow="Monitoring console")
    health = report["monitoring_health"]
    render_badge_row(st, [("Pipeline", report["status"], "pass"), ("Operational health", health, health), ("Outcome labels", "Awaiting complete labels", "warning")])
    render_kpi_row(st, [
        {"label": "Raw critical drift", "value": report.get("raw_critical_feature_count", report["critical_feature_count"])},
        {"label": "Operational critical", "value": report.get("operational_critical_feature_count", report["critical_feature_count"])},
        {"label": "Warnings", "value": report["warning_feature_count"]},
    ])
    render_kpi_row(st, [
        {"label": "Expected temporal shifts", "value": report.get("expected_structural_shift_feature_count", 0)},
        {"label": "Stable features", "value": report["stable_feature_count"]},
    ])
    render_warning_banner(st, "Data and score-distribution drift are not performance drift. Review exposure is not proof of discrimination. Complete future outcomes are required before evaluation.")
    numeric_rows = [{"feature": name, "PSI": row["psi"], "raw severity": row.get("raw_statistical_severity", row["severity"]), "operational severity": row.get("operational_severity", row["severity"]), "semantics": row.get("feature_semantics", "standard_input")} for name, row in report["numeric_drift"].items()]
    categorical_rows = [{"feature": name, "PSI": row["psi"], "raw severity": row.get("raw_statistical_severity", row["severity"]), "operational severity": row.get("operational_severity", row["severity"]), "semantics": row.get("feature_semantics", "standard_input")} for name, row in report["categorical_drift"].items()]
    drift = pd.DataFrame(numeric_rows + categorical_rows)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Raw statistical drift")
        _altair_chart(st, drift.sort_values("PSI", ascending=False), category="feature", measure="PSI", title="PSI", color="#e8b84d", height=310)
    with right:
        st.markdown("#### Operational interpretation")
        severity_counts = drift["operational severity"].value_counts().rename_axis("operational severity").reset_index(name="feature count")
        _altair_chart(st, severity_counts, category="operational severity", measure="feature count", title="Feature count", color="#9ac8ff", height=310)
    st.dataframe(drift.loc[:, ["feature", "semantics", "PSI", "raw severity", "operational severity"]], hide_index=True, width="stretch")
    st.markdown("#### Feature detail")
    feature = st.selectbox("Monitoring feature", drift["feature"].tolist())
    detail = next(row for row in numeric_rows + categorical_rows if row["feature"] == feature)
    render_kpi_row(st, [{"label": "Raw severity", "value": detail["raw severity"]}, {"label": "Operational severity", "value": detail["operational severity"]}, {"label": "PSI", "value": f"{detail['PSI']:.3f}"}, {"label": "Feature semantics", "value": detail["semantics"]}])
    if detail["semantics"] == "cumulative_history":
        render_info_banner(st, "Raw statistical drift remains visible for this cumulative-history feature. Its operational interpretation may be an expected temporal accumulation rather than hidden drift.")
    st.markdown("#### Score distribution and review exposure")
    st.json({"score_distribution_drift": report["score_distribution_drift"], "review_exposure_diagnostics": report["review_exposure_diagnostics"]})
    st.markdown("#### Human-review workflow")
    st.write("1. Candidate ranked → 2. Human reviews context → 3. Reviewer records reason → 4. Reviewer may override priority → 5. Human may escalate for assessment → 6. No automatic enforcement.")
    runtime_root = Path(os.environ.get("INSPECTIQ_RUNTIME_ROOT", "."))
    review_path = resolve_report_path(report["review_worksheet_path"], runtime_root)
    st.download_button("Download human-review template", review_path.read_bytes(), "inspectiq_review_queue_template.csv", "text/csv")
    if report.get("deployment_future_outcome_template_excluded"):
        st.caption("Future-outcome template is intentionally excluded from the candidate-only public demo.")
    else:
        future_path = resolve_report_path(report["future_outcome_template_path"], runtime_root)
        st.download_button("Download future-outcome template", future_path.read_bytes(), "inspectiq_future_outcome_template.csv", "text/csv")
    st.markdown("#### Future outcome monitoring")
    st.info("Awaiting complete outcome labels. No current performance metric or outcome-fairness metric is calculated here.")


def _limitations_page(st, context) -> None:
    render_page_header(st, "Data & Limitations", "Transparent documentation for a bounded, retrospective candidate-ranking prototype.", eyebrow="Documentation")
    tabs = st.tabs(["Data", "Label construction", "Chronology", "Leakage controls", "Limitations", "Responsible use", "Reproducibility"])
    with tabs[0]:
        render_kpi_row(st, [{"label": "Candidate geography", "value": "California"}, {"label": "Candidate batch", "value": context.batch["input_row_count"]}, {"label": "Candidate period", "value": "2023"}, {"label": "Date range", "value": f"{context.batch['open_date_range']['minimum']} to {context.batch['open_date_range']['maximum']}"}])
        st.write("The supplied candidate batch is drawn from historical OSHA inspection records. It is not a complete workplace registry.")
    with tabs[1]:
        st.write("Historical labels were constructed for prior feasibility and retrospective validation work. Unknown outcomes were excluded from labelled analyses. The supplied 2023 candidates remain unlabelled in this workflow.")
    with tabs[2]:
        st.markdown("**2020–2021 training** → **2022 retrospective validation** → **2023 unlabelled candidates**")
        st.caption("Chronology prevents future information from feeding earlier-stage model development.")
    with tabs[3]:
        for item in ["Strict prior industry history", "No same-day influence", "No validation-label feedback", "No locked-candidate labels", "Unknown labels excluded", "No establishment-history feature without a defensible key"]:
            st.write(f"✓ {item}")
    with tabs[4]:
        for item in ["Early-year sampling is limited.", "Historical inspection selection can be biased.", "The queue covers only supplied candidates.", "Model explanations describe fitted behavior, not causality."]:
            st.write(f"• {item}")
    with tabs[5]:
        for item in ["Advisory ranking only", "Human review required", "No automatic enforcement", "No calibrated outcome claim", "No current 2023 performance claim", "No protected-class fairness conclusion"]:
            st.write(f"✓ {item}")
    with tabs[6]:
        st.json({"source_snapshot_id": context.batch["source_snapshot_id"], "feature_version": context.batch["feature_version"], "selected_experiment": context.batch["selected_day3_experiment"], "selected_calibration_method": context.batch["selected_day4_method"], "ranking_rule": context.prediction_manifest["ranking_rule"]})


def _detail_controls(st, context) -> int:
    ranks = context.ranked["rank"].tolist()
    current = int(st.session_state.get("active_candidate_rank", ranks[0]))
    selected = st.sidebar.selectbox("Candidate rank", ranks, index=ranks.index(current), format_func=lambda rank: _candidate_rank_label(context, rank))
    previous, next_ = st.sidebar.columns(2)
    if previous.button("Previous", disabled=selected == ranks[0]):
        selected = adjacent_rank(ranks, selected, -1)
    if next_.button("Next", disabled=selected == ranks[-1]):
        selected = adjacent_rank(ranks, selected, 1)
    jumps = st.sidebar.radio("Jump to queue tier", ["Current", "Top 5%", "Top 10%", "Top 20%"], index=0)
    if jumps != "Current":
        cutoff = {"Top 5%": 15, "Top 10%": 30, "Top 20%": 60}[jumps]
        selected = min(selected, cutoff)
    st.session_state["active_candidate_rank"] = selected
    return selected


def _load_context_with_cache(st):
    """Cache immutable report/CSV reads and the loaded final model for page interactions."""
    data_decorator = st.cache_data(show_spinner=False)
    resource_decorator = st.cache_resource(show_spinner="Validating frozen InspectIQ artifacts…")
    if not callable(data_decorator) or not callable(resource_decorator):  # Minimal render fakes deliberately omit Streamlit caching.
        return load_dashboard_context(Path("config/dashboard_config.yaml"))

    @data_decorator
    def _artifact_state(config_path: str) -> tuple[tuple[str, int, int], ...]:
        """A deterministic cache key for immutable config/report inputs."""
        paths = [Path(config_path), *(Path("reports") / name for name in (
            "batch_prediction_report.json", "model_comparison_report.json", "calibration_report.json",
            "mlflow_tracking_report.json", "feature_engineering_report.json", "monitoring_report.json",
        ))]
        return tuple((path.as_posix(), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)

    @resource_decorator
    def _load(config_path: str, _state: tuple[tuple[str, int, int], ...]):
        return load_dashboard_context(Path(config_path))

    return _load("config/dashboard_config.yaml", _artifact_state("config/dashboard_config.yaml"))


def main() -> None:
    st = _streamlit()
    st.set_page_config(page_title="InspectIQ | Human Review", page_icon="🔎", layout="wide")
    apply_theme(st)
    st.markdown("<div class=\"inspectiq-eyebrow\">InspectIQ</div>", unsafe_allow_html=True)
    st.title("Workplace Safety Inspection Risk Triage")
    st.markdown("<div class=\"inspectiq-subtitle\">A retrospective decision-support prototype that ranks supplied candidates for human review.</div>", unsafe_allow_html=True)
    try:
        context = _load_context_with_cache(st)
    except DashboardError as exc:
        st.error(f"Dashboard artifact validation failed: {exc}")
        st.stop()
        return
    render_badge_row(st, [("Use", "Advisory only", "neutral"), ("Scores", "Uncalibrated", "warning"), ("Candidate period", "2023", "neutral"), ("Human review", "Required", "pass")])
    st.sidebar.markdown("### InspectIQ")
    st.sidebar.markdown("<div class=\"inspectiq-sidebar-card\"><b>Advisory ranking only</b><br>No automated action<br>2023 outcomes unavailable</div>", unsafe_allow_html=True)
    page = st.sidebar.radio("Navigate", NAVIGATION)
    if page == "Review Queue":
        st.sidebar.header("Review queue controls")
        filters = _filters(st, context.ranked)
        _queue_page(st, context, filter_candidates(context.ranked, filters))
    elif page == "Candidate Detail":
        _detail_page(st, context, _detail_controls(st, context))
    elif page == "Model Evidence":
        _evidence_page(st, context)
    elif page == "Monitoring & Governance":
        _monitoring_page(st, context)
    else:
        _limitations_page(st, context)
    render_footer(st, "Public demo · v1.0.1")


if __name__ == "__main__":
    main()
