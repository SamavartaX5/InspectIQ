import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.batch_prediction import MODEL_COLUMNS, OUTPUT_COLUMNS
from src.dashboard import DashboardError, filter_candidates, load_dashboard_context, public_demo_version, queue_csv, validate
from src.explanations import global_feature_importance, local_perturbation_explanation, training_references
from src.ui_components import is_long_metric_value, render_footer, render_kpi_row


class FrozenPreprocess:
    def get_feature_names_out(self):
        return np.asarray([
            "counts__nr_in_estab", "counts__industry_prior_inspection_count", "counts__industry_prior_positive_count",
            "numeric__open_month", "numeric__industry_prior_positive_rate_smoothed",
            "categorical__naics_group_23", "categorical__insp_type_A", "categorical__insp_scope_C",
            "categorical__owner_type_P", "categorical__safety_hlth_S", "categorical__industry_history_status_industry_history",
        ])


class FrozenEstimator:
    feature_importances_ = np.asarray([.05, .10, .15, .05, .05, .10, .10, .10, .10, .10, .10])


class FrozenPipeline:
    feature_names_in_ = np.asarray(MODEL_COLUMNS)
    classes_ = np.asarray([0, 1])
    named_steps = {"preprocess": FrozenPreprocess(), "model": FrozenEstimator()}

    def fit(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("Dashboard must not fit the final model")

    def partial_fit(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("Dashboard must not partial_fit the final model")

    def predict_proba(self, frame):
        score = np.clip(np.asarray(frame["nr_in_estab"], dtype=float) / 20, 0, 1)
        return np.column_stack([1 - score, score])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DashboardTests(unittest.TestCase):
    def fixture(self, root):
        model_path = root / "final_candidate.joblib"; joblib.dump(FrozenPipeline(), model_path)
        candidate = pd.DataFrame({
            "rank": [1, 2, 3, 4], "activity_nr": ["a", "b", "c", "d"], "open_date": ["2023-01-03"] * 4,
            "raw_risk_score": [.8, .5, .5, .2], "score_percentile": [100., 75., 50., 25.],
            "review_priority": ["highest_priority", "high_priority", "high_priority", "standard_priority"],
            "top_5_percent_flag": [True, False, False, False], "top_10_percent_flag": [True, True, True, False],
            "top_20_percent_flag": [True, True, True, False], "naics_group": ["23", "31", "23", "31"],
            "insp_type": ["A", "B", "A", "B"], "insp_scope": ["C"] * 4, "owner_type": ["P"] * 4,
            "safety_hlth": ["S"] * 4, "nr_in_estab": [16, 10, 10, 4], "open_month": [1, 1, 1, 1],
            "industry_prior_inspection_count": [10] * 4, "industry_prior_positive_count": [2] * 4,
            "industry_prior_positive_rate_smoothed": [.2] * 4, "industry_history_status": ["industry_history"] * 4,
        }).loc[:, OUTPUT_COLUMNS]
        output = root / "prediction"; output.mkdir(); ranked, top = output / "ranked_candidates.csv", output / "top_10_percent_candidates.csv"
        candidate.to_csv(ranked, index=False); candidate.head(3).to_csv(top, index=False)
        prediction_manifest = {"prediction_run_id":"run", "source_snapshot_id":"snapshot", "feature_version":"v1", "model_artifact_hash":digest(model_path), "input_feature_hash":"input", "output_hashes":{ranked.name:digest(ranked), top.name:digest(top)}, "ranking_rule":"score descending then activity ascending"}
        write_json(output / "prediction_manifest.json", prediction_manifest)
        training = pd.DataFrame({column: [10, 20, 20] if column == "nr_in_estab" else ["A", "A", "B"] if column in {"insp_type"} else [1, 2, 3] if column in {"open_month", "industry_prior_inspection_count", "industry_prior_positive_count"} else [.1, .2, .3] if column == "industry_prior_positive_rate_smoothed" else ["x", "x", "y"] for column in MODEL_COLUMNS})
        training["serious_violation_found"] = [0, 1, 0]
        training_path = root / "train_features.csv"; training.to_csv(training_path, index=False)
        feature_manifest = root / "feature_manifest.json"; write_json(feature_manifest, {"file_hashes":{training_path.name:digest(training_path)}})
        feature_report = root / "feature.json"; write_json(feature_report, {"status":"PASS", "source_snapshot_id":"snapshot", "feature_version":"v1", "output_paths":{"train":str(training_path), "manifest":str(feature_manifest)}, "splits":{"test_locked":{"target_present":False}}})
        batch = root / "batch.json"; write_json(batch, {"status":"PASS", "source_snapshot_id":"snapshot", "feature_version":"v1", "prediction_run_id":"run", "model_artifact_hash":digest(model_path), "model_artifact_path":str(model_path), "locked_candidate_input_hash":"input", "ranked_output_path":str(ranked), "top_10_output_path":str(top), "top_counts":{"top_10_percent":3}, "review_priority_counts":{"highest_priority":1,"high_priority":2,"standard_priority":1}, "selected_day3_experiment":"rf", "selected_day4_method":"uncalibrated", "open_date_range":{}, "input_row_count":4, "labels_accessed":False, "performance_metrics_calculated":False, "evaluation_performed":False, "automatic_enforcement":False})
        comparison = root / "comparison.json"; write_json(comparison, {"status":"PASS", "source_snapshot_id":"snapshot", "feature_version":"v1", "selected_candidate":"rf", "baseline":{"positives_captured_top_10":1}, "experiments":[{"experiment_id":"rf", "metrics":{"ranking_at":{"10":{"selected_positives":2,"recall":.2,"precision":.3,"lift":1.2}},"pr_auc":.2,"roc_auc":.6,"brier_score":.2}}]})
        calibration = root / "calibration.json"; write_json(calibration, {"status":"PASS", "source_snapshot_id":"snapshot", "feature_version":"v1", "final_candidate_artifact_path":str(model_path), "selected_calibration_method":"uncalibrated", "selection_rationale":"calibration rejected", "locked_test_labels_accessed":False, "locked_test_metrics_calculated":False, "locked_test_predictions_created":False})
        mlflow = root / "mlflow.json"; write_json(mlflow, {"status":"PASS", "logged_day3_run_count":8, "logged_day4_run_count":3})
        config = root / "config.yaml"; config.write_text("expected_ranked_rows: 4\nexpected_top_10_rows: 3\nreports:\n  batch_prediction: " + str(batch).replace("\\", "/") + "\n  model_comparison: " + str(comparison).replace("\\", "/") + "\n  calibration: " + str(calibration).replace("\\", "/") + "\n  mlflow_tracking: " + str(mlflow).replace("\\", "/") + "\n  feature_engineering: " + str(feature_report).replace("\\", "/") + "\n", encoding="utf8")
        return config, ranked, candidate

    def test_contract_validation_and_offline_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _ranked, _candidate = self.fixture(Path(temp))
            context = load_dashboard_context(config); report = validate(config)
        self.assertEqual(len(context.ranked), 4); self.assertEqual(len(context.top_10), 3)
        self.assertTrue(report["required_column_checks"]); self.assertFalse(report["locked_labels_accessed"])
        self.assertFalse(report["performance_metrics_calculated"]); self.assertFalse(report["automatic_enforcement"])
        self.assertAlmostEqual(report["global_importance_sum"], 1.0)

    def test_contract_rejects_ids_ranks_scores_target_and_outcomes(self):
        changes = [
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            lambda frame: frame.assign(rank=[1, 3, 4, 5]),
            lambda frame: frame.assign(raw_risk_score=[1.1, .5, .5, .2]),
            lambda frame: frame.assign(serious_violation_found=0),
            lambda frame: frame.assign(current_penalty=1),
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                config, ranked_path, frame = self.fixture(Path(temp)); altered = change(frame); altered.to_csv(ranked_path, index=False)
                manifest = ranked_path.parent / "prediction_manifest.json"; value = json.loads(manifest.read_text()); value["output_hashes"][ranked_path.name] = digest(ranked_path); write_json(manifest, value)
                batch_path = Path(yaml_path(config, "batch_prediction")); batch = json.loads(batch_path.read_text()); batch["review_priority_counts"] = {key:int(value) for key,value in altered["review_priority"].value_counts().to_dict().items()} if "review_priority" in altered else {}; write_json(batch_path, batch)
                with self.assertRaises(DashboardError): load_dashboard_context(config)

    def test_filtering_downloads_and_explanations_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _ranked, _candidate = self.fixture(Path(temp)); context = load_dashboard_context(config)
            filtered = filter_candidates(context.ranked, {"naics_group":"23", "insp_type":"A", "raw_risk_score":(.4, .9)})
            empty = filter_candidates(context.ranked, {"owner_type":"missing"})
            references = training_references(context.training); first = local_perturbation_explanation(context.model, context.ranked.iloc[0], references); second = local_perturbation_explanation(context.model, context.ranked.iloc[0], references)
            importance = global_feature_importance(context.model)
        self.assertEqual(filtered.activity_nr.tolist(), ["a", "c"]); self.assertTrue(empty.empty)
        self.assertEqual(queue_csv(filtered), queue_csv(filtered)); self.assertNotIn("serious_violation_found", queue_csv(filtered).decode())
        self.assertEqual(first.to_dict("records"), second.to_dict("records")); self.assertEqual(len(first), len(MODEL_COLUMNS))
        self.assertEqual(references["nr_in_estab"], 20.0); self.assertEqual(references["insp_type"], "A")
        self.assertAlmostEqual(importance.importance.sum(), 1.0); self.assertTrue(importance.importance.is_monotonic_decreasing)

    def test_app_import_and_failure_preserves_report(self):
        import app.streamlit_app as app
        import run_dashboard_validation
        self.assertIn("Review Queue", app.NAVIGATION); self.assertIn("Monitoring & Governance", app.NAVIGATION); self.assertIn("uncalibrated", app.SCORE_CAVEAT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); reports = root / "reports"; reports.mkdir(); valid = reports / "dashboard_validation_report.json"; valid.write_text('{"status":"PASS","keep":true}', encoding="utf8")
            old = Path.cwd(); os.chdir(root)
            try:
                with patch("run_dashboard_validation.validate", side_effect=DashboardError("bad artifact")):
                    run_dashboard_validation.main()
            finally:
                os.chdir(old)
            self.assertIn("keep", valid.read_text()); self.assertIn("bad artifact", (reports / "dashboard_validation_attempt_error.json").read_text())

    def test_review_queue_main_render_smoke(self):
        import app.streamlit_app as app
        class FakeColumnConfig:
            class NumberColumn:
                def __init__(self, **_kwargs): pass
        class FakeSt:
            column_config = FakeColumnConfig()
            def __init__(self): self.sidebar = self; self.session_state = {}
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def button(self, *_args, **_kwargs): return False
            def multiselect(self, *_args, **_kwargs): return []
            def slider(self, _label, **kwargs): return kwargs["value"]
            def radio(self, *_args, **_kwargs): return "Review Queue"
            def columns(self, count): return [self] * count
            def cache_data(self, **_kwargs): return lambda function: function
            def cache_resource(self, **_kwargs): return lambda function: function
            def __getattr__(self, _name): return lambda *_args, **_kwargs: None
        with tempfile.TemporaryDirectory() as temp:
            config, _ranked, _candidate = self.fixture(Path(temp)); context = load_dashboard_context(config)
            with patch("app.streamlit_app._streamlit", return_value=FakeSt()), patch("app.streamlit_app.load_dashboard_context", return_value=context):
                app.main()

    def test_candidate_detail_uses_rank_column_not_dataframe_rank_method(self):
        import app.streamlit_app as app
        class FakeSt:
            def __init__(self): self.options = None
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def selectbox(self, _label, options, **_kwargs): self.options = options; return options[0]
            def columns(self, count): return [self] * count
            def __getattr__(self, _name): return lambda *_args, **_kwargs: None
        with tempfile.TemporaryDirectory() as temp:
            config, _ranked, _candidate = self.fixture(Path(temp)); context = load_dashboard_context(config); st = FakeSt()
            app._detail_page(st, context)
        self.assertEqual(st.options, [1, 2, 3, 4])

    def test_review_budget_filter_and_navigation_helpers_preserve_frozen_values(self):
        import app.streamlit_app as app
        frame = pd.DataFrame({"rank": list(range(1, 301)), "raw_risk_score": [number / 300 for number in range(1, 301)], "review_priority": ["standard_priority"] * 300, "naics_group": ["23"] * 300, "insp_type": ["A"] * 300, "insp_scope": ["A"] * 300, "owner_type": ["P"] * 300, "safety_hlth": ["S"] * 300})
        self.assertEqual([len(app.budget_queue(frame, value)) for value in ("5%", "10%", "20%")], [15, 30, 60])
        filtered = filter_candidates(frame, {"naics_group": "23", "raw_risk_score": (0.2, 0.8)})
        self.assertEqual(filtered["rank"].tolist(), frame.loc[frame["raw_risk_score"].between(.2, .8), "rank"].tolist())
        self.assertEqual(app.adjacent_rank([1, 2, 3], 1, -1), 1)
        self.assertEqual(app.adjacent_rank([1, 2, 3], 2, 1), 3)
        self.assertEqual(app.active_filter_count({"naics_group": ["23"]}, frame), 1)

    def test_altair_bar_marks_are_explicitly_visible_for_every_dashboard_chart(self):
        import app.streamlit_app as app
        with tempfile.TemporaryDirectory() as temp:
            config, _ranked, candidate = self.fixture(Path(temp))
            context = load_dashboard_context(config)
            queue_frames = app.review_queue_chart_frames(context.ranked, app.budget_queue(context.ranked, "All candidates"))
        self.assertEqual(set(queue_frames), {"queue_priority", "naics_mix", "score_distribution", "inspection_type"})
        self.assertTrue(all(not frame.empty for frame in queue_frames.values()))
        chart_inputs = [
            (queue_frames["queue_priority"], "review priority", "candidate count", app.CHART_COLORS["queue_priority"]),
            (queue_frames["naics_mix"], "NAICS group", "candidate count", app.CHART_COLORS["naics_mix"]),
            (queue_frames["score_distribution"], "score band", "candidate count", app.CHART_COLORS["score_distribution"]),
            (queue_frames["inspection_type"], "inspection type", "candidate count", app.CHART_COLORS["inspection_type"]),
        ]
        generic = pd.DataFrame({"category": ["one", "two"], "measure": [1, 2]})
        chart_inputs.extend((generic, "category", "measure", color) for color in app.CHART_COLORS.values())
        for frame, category, measure, color in chart_inputs:
            with self.subTest(color=color, category=category):
                self.assertTrue(app._valid_hex_color(color))
                spec = app._build_bar_chart(frame, category=category, measure=measure, title="Count", color=color, height=220).to_dict()
                mark = spec["mark"]
                self.assertEqual(mark["type"], "bar")
                self.assertEqual(mark["color"], color)
                self.assertEqual(mark["fill"], color)
                self.assertGreater(mark["opacity"], 0)
                self.assertNotEqual(mark["color"].lower(), app.CHART_BACKGROUND.lower())
                self.assertNotEqual(mark["fill"].lower(), app.CHART_BACKGROUND.lower())
                self.assertNotIn("var(", mark["color"].lower())
                self.assertNotIn("var(", mark["fill"].lower())
                self.assertTrue(app._valid_hex_color(mark["stroke"]))
                self.assertNotEqual(mark["stroke"].lower(), app.CHART_BACKGROUND.lower())

    def test_public_ui_contract_is_safe_and_theme_is_local(self):
        import app.streamlit_app as app
        import tomllib
        source = Path("app/streamlit_app.py").read_text(encoding="utf8")
        theme = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf8"))
        css = Path("src/ui_theme.py").read_text(encoding="utf8").lower()
        self.assertEqual(app.NAVIGATION, ["Review Queue", "Candidate Detail", "Model Evidence", "Monitoring & Governance", "Data & Limitations"])
        self.assertIn("uncalibrated advisory score", app.SCORE_CAVEAT)
        self.assertIn("No automatic enforcement", source)
        self.assertNotIn("use_container_width", source)
        self.assertNotIn("requests.", source)
        self.assertNotIn("<script", css)
        self.assertNotIn("fonts.googleapis", css)
        self.assertEqual(theme["theme"]["base"], "dark")

    def test_long_metric_values_wrap_without_truncation_and_footer_uses_deployment_version(self):
        import app.streamlit_app as app
        date_range = "2023-01-03 to 2023-01-23"

        class Column:
            def __init__(self): self.markdown_calls = []; self.metric_calls = []
            def markdown(self, value, **_kwargs): self.markdown_calls.append(value)
            def metric(self, *args, **kwargs): self.metric_calls.append((args, kwargs))
            def caption(self, *_args, **_kwargs): pass

        class FakeSt:
            def __init__(self): self.columns_created = []; self.markdown_calls = []
            def columns(self, count):
                self.columns_created = [Column() for _ in range(count)]
                return self.columns_created
            def markdown(self, value, **_kwargs): self.markdown_calls.append(value)

        st = FakeSt()
        render_kpi_row(st, [{"label": "Date range", "value": date_range}])
        long_card = st.columns_created[0].markdown_calls[0]
        render_footer(st, f"Public demo · {public_demo_version()}")
        theme_source = Path("src/ui_theme.py").read_text(encoding="utf8")
        dashboard_source = Path("app/streamlit_app.py").read_text(encoding="utf8")
        dashboard_config = Path("config/dashboard_config.yaml").read_text(encoding="utf8")
        deployment_config = Path("config/deployment_config.yaml").read_text(encoding="utf8")
        self.assertTrue(is_long_metric_value(date_range))
        self.assertIn(date_range, long_card)
        self.assertNotIn("...", long_card)
        self.assertIn("inspectiq-metric-card--long", long_card)
        self.assertIn("overflow-wrap: anywhere", theme_source)
        self.assertIn("white-space: normal", theme_source)
        self.assertIn("text-overflow: clip", theme_source)
        self.assertEqual(public_demo_version(), "v1.1.1")
        self.assertTrue(any("Public demo · v1.1.1" in value for value in st.markdown_calls))
        self.assertNotIn("v1.0.1", dashboard_source)
        self.assertNotIn("v1.0.1", dashboard_config)
        self.assertNotIn("v1.0.1", deployment_config)
        self.assertEqual(app.NAVIGATION, ["Review Queue", "Candidate Detail", "Model Evidence", "Monitoring & Governance", "Data & Limitations"])


def yaml_path(config, key):
    import yaml
    return yaml.safe_load(Path(config).read_text())["reports"][key]


if __name__ == "__main__":
    unittest.main()
