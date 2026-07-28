"""Focused, synthetic tests for the read-only Day 7A release validator."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.governance import REVIEW_COLUMNS, future_outcome_template
from src.release_validation import ReleaseValidationError, validate_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseValidationTests(unittest.TestCase):
    def _write(self, root: Path, name: str, text: str) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _fixture(self, *, local: bool = False) -> Path:
        temporary = Path(tempfile.mkdtemp())
        config = yaml.safe_load((PROJECT_ROOT / "config" / "release_config.yaml").read_text(encoding="utf-8"))
        self._write(temporary, "config/release_config.yaml", yaml.safe_dump(config, sort_keys=False))
        self._write(temporary, "config/dashboard_config.yaml", "dashboard: true\n")
        self._write(temporary, "config/monitoring_config.yaml", "monitoring: true\n")
        for name in ("app/streamlit_app.py", "src/governance.py", "src/monitoring.py", "src/release_validation.py", "run_release_validation.py"):
            self._write(temporary, name, '"""human review; no outcomes are created."""\n')
        self._write(
            temporary,
            "Dockerfile",
            "FROM python:3.13-slim\nENV PYTHONPATH=/app\nUSER appuser\nEXPOSE 8501\nCMD [\"python\", \"-m\", \"streamlit\", \"run\", \"app/streamlit_app.py\", \"--server.address=0.0.0.0\"]\n",
        )
        self._write(temporary, ".dockerignore", ".env\nartifacts\ndata/raw\nmlflow.db\nmlartifacts\n")
        self._write(
            temporary,
            ".github/workflows/ci.yml",
            "name: CI\non:\n  push:\n  pull_request:\n  workflow_dispatch:\npermissions:\n  contents: read\njobs:\n  verify:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.13'\n      - run: python -m compileall app src scripts tests *.py\n      - run: python -m unittest discover -s tests -t . -v\n      - run: python run_release_validation.py --mode ci\n      - run: docker build .\n",
        )
        names = config["required_committed_reports"]
        reports = {name: {"status": "PASS"} for name in names}
        reports["reports/feasibility_report.json"] = {"decision": "GO"}
        if local:
            reports.update(self._local_reports(temporary))
        for name, report in reports.items():
            self._write(temporary, name, json.dumps(report))
        self.addCleanup(lambda: __import__("shutil").rmtree(temporary, ignore_errors=True))
        return temporary

    def _local_reports(self, root: Path) -> dict[str, dict]:
        snapshot, version = "snapshot-1", "day2-historical-v1"
        model = self._write(root, "artifacts/models/final_candidate.joblib", "frozen model")
        ranked = root / "artifacts/predictions/ranked.csv"
        top = root / "artifacts/predictions/top.csv"
        ranked.parent.mkdir(parents=True, exist_ok=True)
        headers = ["activity_nr", "advisory_score", "rank"]
        with ranked.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers); writer.writeheader()
            for number in range(300): writer.writerow({"activity_nr": str(number), "advisory_score": "0.2", "rank": str(number + 1)})
        with top.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers); writer.writeheader()
            for number in range(30): writer.writerow({"activity_nr": str(number), "advisory_score": "0.2", "rank": str(number + 1)})
        monitoring = root / "artifacts/monitoring"; monitoring.mkdir(parents=True, exist_ok=True)
        manifest = self._write(root, "artifacts/monitoring/monitoring_manifest.json", "{}")
        template = monitoring / "future_outcome_template.csv"; future_outcome_template().to_csv(template, index=False)
        worksheet = monitoring / "review_queue_template.csv"
        with worksheet.open("w", newline="", encoding="utf-8") as handle: csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS).writeheader()
        shared = {"source_snapshot_id": snapshot, "feature_version": version}
        return {
            "reports/feature_engineering_report.json": {"status": "PASS", **shared},
            "reports/model_comparison_report.json": {"status": "PASS", **shared, "selected_candidate": {"experiment_name": "exp_05_random_forest"}},
            "reports/calibration_report.json": {"status": "PASS", **shared, "selected_calibration_method": "uncalibrated", "final_candidate_artifact_path": "artifacts/models/final_candidate.joblib"},
            "reports/mlflow_tracking_report.json": {"status": "PASS", "logged_day3_run_count": 8, "logged_day4_run_count": 3, "final_candidate_artifact_logged": True, "reused_run_count": 11, "newly_created_run_count": 0},
            "reports/batch_prediction_report.json": {"status": "PASS", **shared, "selected_day3_experiment": "exp_05_random_forest", "selected_day4_method": "uncalibrated", "model_artifact_hash": _digest(model), "ranked_output_path": "artifacts/predictions/ranked.csv", "ranked_output_hash": _digest(ranked), "top_10_output_path": "artifacts/predictions/top.csv", "top_10_output_hash": _digest(top), "labels_accessed": False, "performance_metrics_calculated": False, "automatic_enforcement": False},
            "reports/dashboard_validation_report.json": {"status": "PASS", **shared, "model_refit_attempted": False},
            "reports/monitoring_report.json": {"status": "PASS", **shared, "monitoring_health": "WARNING", "monitoring_manifest_path": "artifacts/monitoring/monitoring_manifest.json", "future_outcome_template_path": "artifacts/monitoring/future_outcome_template.csv", "future_outcome_template_hash": _digest(template), "review_worksheet_path": "artifacts/monitoring/review_queue_template.csv", "review_worksheet_hash": _digest(worksheet), "current_labels_accessed": False, "current_performance_metrics_calculated": False, "outcome_fairness_metrics_calculated": False, "automatic_enforcement": False, "model_refit_attempted": False, "prediction_artifact_modified": False},
        }

    def test_ci_mode_needs_no_generated_artifacts_or_mutation(self) -> None:
        root = self._fixture()
        before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
        report = validate_release(root, "ci")
        after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
        self.assertEqual("PASS", report["status"])
        self.assertEqual("not required in CI mode", report["artifact_checks"]["artifacts"])
        self.assertEqual(before, after)

    def test_ci_rejects_workflow_secret_and_missing_docker_contract(self) -> None:
        root = self._fixture()
        workflow = root / ".github/workflows/ci.yml"
        workflow.write_text(workflow.read_text(encoding="utf-8") + "\n      - run: ${{ secrets.BAD }}\n", encoding="utf-8")
        with self.assertRaisesRegex(ReleaseValidationError, "CI workflow"):
            validate_release(root, "ci")

    def test_ci_reports_invalid_json_clearly(self) -> None:
        root = self._fixture()
        (root / "reports/baseline_report.json").write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ReleaseValidationError, "Invalid JSON report"):
            validate_release(root, "ci")

    def test_ci_requires_python_313_and_required_steps(self) -> None:
        root = self._fixture()
        workflow = root / ".github/workflows/ci.yml"
        workflow.write_text(workflow.read_text(encoding="utf-8").replace("3.13", "3.12"), encoding="utf-8")
        with self.assertRaisesRegex(ReleaseValidationError, "CI workflow"):
            validate_release(root, "ci")

    def test_ci_requires_dockerfile_and_dockerignore(self) -> None:
        root = self._fixture()
        (root / ".dockerignore").unlink()
        with self.assertRaisesRegex(ReleaseValidationError, "Required source files missing"):
            validate_release(root, "ci")

    def test_local_validates_rows_hashes_and_warning_health(self) -> None:
        root = self._fixture(local=True)
        report = validate_release(root, "local")
        self.assertEqual("PASS", report["status"])
        self.assertEqual("WARNING", _read_report(root, "monitoring_report.json")["monitoring_health"])
        self.assertTrue(report["artifact_checks"]["hashes"]["ranked"])

    def test_local_rejects_target_field_and_preserves_frozen_outputs(self) -> None:
        root = self._fixture(local=True)
        ranked = root / "artifacts/predictions/ranked.csv"
        original = ranked.read_text(encoding="utf-8")
        ranked.write_text(original.replace("rank\n", "rank,label\n", 1), encoding="utf-8")
        with self.assertRaises(ReleaseValidationError):
            validate_release(root, "local")
        self.assertEqual(original.replace("rank\n", "rank,label\n", 1), ranked.read_text(encoding="utf-8"))

    def test_local_rejects_mismatched_snapshot_and_safety_flag(self) -> None:
        root = self._fixture(local=True)
        report = _read_report(root, "batch_prediction_report.json")
        report["source_snapshot_id"] = "different"
        (root / "reports/batch_prediction_report.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ReleaseValidationError, "Source snapshot IDs"):
            validate_release(root, "local")


def _read_report(root: Path, name: str) -> dict:
    return json.loads((root / "reports" / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
