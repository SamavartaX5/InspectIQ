import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.mlflow_tracking import TrackingError, run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def ranking():
    return {str(p): {"recall": .1, "precision": .2, "lift": 1.2, "selected_positives": 2} for p in (5, 10, 20)}


class MlflowTrackingTests(unittest.TestCase):
    def fixture(self, root):
        artifacts = root / "artifacts"
        day3_pipe, day3_score = artifacts / "model.joblib", artifacts / "validation_scoring.csv"
        study = {name: artifacts / f"{name}.joblib" for name in ("uncalibrated", "sigmoid", "isotonic")}
        final = artifacts / "final_candidate.joblib"
        for path in [day3_pipe, day3_score, *study.values(), final]:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"artifact")
        metrics = {"ranking_at": ranking(), "pr_auc": .3, "roc_auc": .6, "brier_score": .2, "threshold_0_5": {"precision": .2, "recall": .3, "f1": .25}}
        day3 = {"status":"PASS","source_snapshot_id":"s","feature_version":"f","training_run_id":"t","train_shape":[10,3],"validation_shape":[8,3],"selected_candidate":"e1","ml_value_added":True,"locked_test_labels_accessed":False,"locked_test_metrics_calculated":False,"experiments":[{"experiment_id":"e1","model_name":"random_forest","hyperparameters":{"random_state":42},"metrics":metrics,"training_runtime_seconds":1.,"validation_scoring_runtime_seconds":.1,"artifact_size_bytes":8,"artifact_path":str(day3_pipe)}]}
        methods = {name:{**metrics,"log_loss":.4,"expected_calibration_error":.1,"maximum_calibration_error":.2,"mean_predicted_probability":.2,"observed_positive_rate":.2,"mean_probability_gap":0.} for name in study}
        day4 = {"status":"PASS","source_snapshot_id":"s","feature_version":"f","calibration_run_id":"c","selected_day3_experiment":"e1","selected_calibration_method":"uncalibrated","calibration_improved_probability_quality":False,"final_calibration_applied":False,"final_candidate_artifact_path":str(final),"locked_test_labels_accessed":False,"locked_test_metrics_calculated":False,"locked_test_predictions_created":False,"study_results":methods,"study_artifact_paths":{k:str(v) for k,v in study.items()}}
        d3, d4, cfg = root / "day3.json", root / "day4.json", root / "config.json"; write(d3,day3); write(d4,day4); write(cfg,{"tracking_uri":f"sqlite:///{root/'tracking.db'}","artifact_root":str(root/'mlartifacts'),"day3_experiment":"D3","day4_experiment":"D4"})
        return d3,d4,cfg

    def test_all_runs_mapping_tags_artifacts_and_idempotency(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            d3,d4,cfg=self.fixture(Path(temp)); first=run(d3,d4,cfg); second=run(d3,d4,cfg)
        self.assertEqual(first["expected_day3_run_count"],1); self.assertEqual(first["expected_day4_run_count"],3)
        self.assertEqual(first["newly_created_run_count"],4); self.assertEqual(second["newly_created_run_count"],0); self.assertEqual(second["reused_run_count"],4)
        self.assertEqual(first["tracking_batch_id"],second["tracking_batch_id"]); self.assertTrue(first["final_candidate_artifact_logged"])

    def test_rejects_nonfinite_incompatible_and_missing_artifacts(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root=Path(temp); d3,d4,cfg=self.fixture(root); value=json.loads(d3.read_text()); value["experiments"][0]["metrics"]["pr_auc"]=math.nan; write(d3,value)
            with self.assertRaises(TrackingError): run(d3,d4,cfg)
            self.fixture(root); value=json.loads(d4.read_text()); value["feature_version"]="other"; write(d4,value)
            with self.assertRaises(TrackingError): run(d3,d4,cfg)
            self.fixture(root); value=json.loads(d4.read_text()); Path(value["final_candidate_artifact_path"]).unlink()
            with self.assertRaises(TrackingError): run(d3,d4,cfg)

    def test_no_training_or_locked_test_access(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            d3,d4,cfg=self.fixture(Path(temp))
            with patch("src.training.pipeline", side_effect=AssertionError("training invoked")):
                report=run(d3,d4,cfg)
        self.assertFalse(report["locked_test_labels_accessed"]); self.assertFalse(report["locked_test_metrics_calculated"]); self.assertFalse(report["locked_test_predictions_created"])


if __name__ == "__main__":
    unittest.main()
