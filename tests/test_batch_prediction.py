import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.batch_prediction import (
    INPUT_COLUMNS, MODEL_COLUMNS, OUTPUT_COLUMNS, PredictionError, run, sha256,
)


class ReversedClassModel:
    """Small serializable inference-only fixture with positive class at column zero."""
    feature_names_in_ = np.asarray(MODEL_COLUMNS)
    classes_ = np.asarray([1, 0])

    def __init__(self):
        self.fit_calls = 0

    def fit(self, *_args, **_kwargs):  # pragma: no cover - must never be invoked
        self.fit_calls += 1
        raise AssertionError("fit must not be called during batch prediction")

    def partial_fit(self, *_args, **_kwargs):  # pragma: no cover - must never be invoked
        raise AssertionError("partial_fit must not be called during batch prediction")

    def predict_proba(self, frame):
        raw = np.asarray(frame["open_month"], dtype=float) / 20
        return np.column_stack([raw, 1 - raw])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


class BatchPredictionTests(unittest.TestCase):
    def fixture(self, root, rows=300):
        features = root / "features.csv"
        frame = pd.DataFrame({
            "activity_nr": [f"{3000 - index:04d}" for index in range(rows)],
            "open_date": ["2023-01-03"] * rows,
            "naics_group": ["23"] * rows, "insp_type": ["A"] * rows,
            "insp_scope": ["C"] * rows, "owner_type": ["P"] * rows,
            "safety_hlth": ["S"] * rows, "nr_in_estab": [5] * rows,
            "open_month": [(index % 12) + 1 for index in range(rows)],
            "industry_prior_inspection_count": [10] * rows,
            "industry_prior_positive_count": [2] * rows,
            "industry_prior_positive_rate_smoothed": [.2] * rows,
            "industry_history_status": ["industry_history"] * rows,
        })
        frame.to_csv(features, index=False)
        manifest = root / "feature_manifest.json"
        write_json(manifest, {"source_snapshot_id":"snapshot", "feature_version":"v1", "file_hashes":{features.name:sha256(features)}, "output_row_counts":{features.name:rows}, "output_columns":{features.name:INPUT_COLUMNS}})
        report = root / "feature_report.json"
        write_json(report, {"status":"PASS", "source_snapshot_id":"snapshot", "feature_version":"v1", "output_paths":{"test_locked":str(features), "manifest":str(manifest)}, "output_hashes":{features.name:sha256(features)}, "splits":{"test_locked":{"output_row_count":rows}}})
        model = root / "final_candidate.joblib"; joblib.dump(ReversedClassModel(), model)
        calibration = root / "calibration.json"
        write_json(calibration, {"status":"PASS", "source_snapshot_id":"snapshot", "feature_version":"v1", "selected_day3_experiment":"rf", "selected_calibration_method":"uncalibrated", "final_calibration_applied":False, "final_candidate_artifact_path":str(model), "locked_test_labels_accessed":False, "locked_test_metrics_calculated":False, "locked_test_predictions_created":False})
        config = root / "config.yaml"
        config.write_text("prediction_version: test\ntop_fractions:\n  top_5_percent: 0.05\n  top_10_percent: 0.10\n  top_20_percent: 0.20\noutput_root: " + str(root / "predictions").replace("\\", "/") + "\n", encoding="utf8")
        return calibration, report, config, features, frame

    def test_scores_rank_positive_class_and_review_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            calibration, report_path, config, _features, _frame = self.fixture(Path(temp))
            report = run(calibration, report_path, config)
            ranked = pd.read_csv(report["ranked_output_path"])
            top_count = len(pd.read_csv(report["top_10_output_path"]))
        self.assertEqual(len(ranked), 300)
        self.assertEqual(list(ranked.columns), OUTPUT_COLUMNS)
        self.assertEqual(top_count, 30)
        self.assertEqual(report["top_counts"], {"top_5_percent":15, "top_10_percent":30, "top_20_percent":60})
        self.assertTrue(ranked["raw_risk_score"].is_monotonic_decreasing)
        self.assertEqual(ranked.iloc[0]["raw_risk_score"], .6)  # class 1 is column zero, not assumed column one
        self.assertEqual(ranked.loc[14, "review_priority"], "highest_priority")
        self.assertEqual(ranked.loc[15, "review_priority"], "high_priority")
        self.assertEqual(ranked.loc[30, "review_priority"], "elevated_priority")
        self.assertEqual(ranked.loc[60, "review_priority"], "standard_priority")
        self.assertFalse(report["labels_accessed"]); self.assertFalse(report["performance_metrics_calculated"])
        self.assertFalse(report["evaluation_performed"]); self.assertFalse(report["automatic_enforcement"])
        self.assertFalse({"serious_violation_found", "current_penalty", "viol_type"}.intersection(ranked.columns))
        self.assertFalse(any(name in report for name in ("recall", "precision", "lift", "brier_score", "roc_auc")))

    def test_input_contract_rejects_target_outcomes_ids_missing_features_and_schema_changes(self):
        mutations = [
            lambda f: f.assign(serious_violation_found=0),
            lambda f: f.assign(current_penalty=1),
            lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True),
            lambda f: f.assign(activity_nr=[None, *f.activity_nr.iloc[1:].tolist()]),
            lambda f: f.drop(columns=["naics_group"]),
            lambda f: f.assign(extra_column=1),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temp:
                calibration, report_path, config, features, frame = self.fixture(Path(temp), rows=4)
                changed = mutate(frame); changed.to_csv(features, index=False)
                # Update only hashes/counts where necessary: schema/id checks must be reached honestly.
                feature = json.loads(report_path.read_text()); feature["output_hashes"][features.name] = sha256(features); feature["splits"]["test_locked"]["output_row_count"] = len(changed); write_json(report_path, feature)
                manifest_path = Path(feature["output_paths"]["manifest"]); manifest = json.loads(manifest_path.read_text()); manifest["file_hashes"][features.name] = sha256(features); manifest["output_row_counts"][features.name] = len(changed); write_json(manifest_path, manifest)
                with self.assertRaises(PredictionError):
                    run(calibration, report_path, config)

    def test_tie_breaker_determinism_reuse_rebuild_and_hash_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); calibration, report_path, config, _features, _frame = self.fixture(root, rows=4)
            first = run(calibration, report_path, config); ranked_path = Path(first["ranked_output_path"])
            ranked = pd.read_csv(ranked_path)
            ties = ranked[ranked["raw_risk_score"] == ranked["raw_risk_score"].iloc[0]]["activity_nr"].astype(str).tolist()
            self.assertEqual(ties, sorted(ties))
            second = run(calibration, report_path, config)
            self.assertTrue(second["artifacts_reused"]); self.assertEqual(first["prediction_run_id"], second["prediction_run_id"])
            self.assertEqual(first["ranked_output_hash"], second["ranked_output_hash"])
            directory = ranked_path.parent
            manifest = json.loads((directory / "prediction_manifest.json").read_text())
            self.assertEqual(manifest["output_hashes"]["ranked_candidates.csv"], sha256(ranked_path))
            (directory / "top_10_percent_candidates.csv").unlink()
            rebuilt = run(calibration, report_path, config)
            self.assertFalse(rebuilt["artifacts_reused"]); self.assertEqual(first["ranked_output_hash"], rebuilt["ranked_output_hash"])
            Path(rebuilt["ranked_output_path"]).write_text("corrupt", encoding="utf8")
            repaired = run(calibration, report_path, config)
            self.assertFalse(repaired["artifacts_reused"]); self.assertEqual(first["ranked_output_hash"], repaired["ranked_output_hash"])
            self.assertFalse(list((root / "predictions").rglob("*.tmp-*")))

    def test_no_training_or_label_source_is_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            calibration, report_path, config, features, _frame = self.fixture(Path(temp), rows=4)
            original_read_csv = pd.read_csv
            loaded = []
            def record(path, *args, **kwargs):
                loaded.append(str(path)); return original_read_csv(path, *args, **kwargs)
            with patch("src.batch_prediction.pd.read_csv", side_effect=record):
                result = run(calibration, report_path, config)
        self.assertIn(str(features), loaded)
        self.assertTrue(all("label" not in path.lower() and "violation" not in path.lower() for path in loaded))
        self.assertNotIn("label", " ".join(loaded).lower()); self.assertNotIn("violation", " ".join(loaded).lower())

    def test_failed_cli_preserves_existing_report(self):
        import os
        import run_batch_prediction
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); reports = root / "reports"; reports.mkdir(); existing = reports / "batch_prediction_report.json"; existing.write_text('{"status":"PASS","detail":"preserve"}', encoding="utf8")
            old = Path.cwd(); os.chdir(root)
            try:
                with patch("run_batch_prediction.run", side_effect=PredictionError("bad candidate input")):
                    run_batch_prediction.main()
            finally:
                os.chdir(old)
            self.assertIn("preserve", existing.read_text())
            self.assertIn("bad candidate input", (reports / "batch_prediction_attempt_error.json").read_text())


if __name__ == "__main__":
    unittest.main()
