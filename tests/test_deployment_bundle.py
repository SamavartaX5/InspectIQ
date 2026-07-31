import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.deployment_bundle import DeploymentBundleError, validate_bundle


class DeploymentBundleTests(unittest.TestCase):
    def _write(self, root: Path, name: str, contents: str) -> Path:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(contents, encoding="utf-8"); return path

    def _bundle(self) -> Path:
        root = Path(tempfile.mkdtemp()); self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        header = "rank,activity_nr,raw_risk_score\n"
        rows = [f"{number},id{number},{1 - number / 1000:.6f}\n" for number in range(1, 301)]
        ranked = self._write(root, "artifacts/predictions/run/ranked_candidates.csv", header + "".join(rows))
        top = self._write(root, "artifacts/predictions/run/top_10_percent_candidates.csv", header + "".join(rows[:30]))
        prediction = {"output_hashes": {ranked.name: hashlib.sha256(ranked.read_bytes()).hexdigest(), top.name: hashlib.sha256(top.read_bytes()).hexdigest()}}
        self._write(root, "artifacts/predictions/run/prediction_manifest.json", json.dumps(prediction))
        batch = {"ranked_output_path": "artifacts/predictions/run/ranked_candidates.csv", "top_10_output_path": "artifacts/predictions/run/top_10_percent_candidates.csv", "ranked_output_hash": prediction["output_hashes"][ranked.name], "top_10_output_hash": prediction["output_hashes"][top.name]}
        batch_path = self._write(root, "reports/batch_prediction_report.json", json.dumps(batch))
        files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("artifacts/predictions/run/ranked_candidates.csv", "artifacts/predictions/run/top_10_percent_candidates.csv", "artifacts/predictions/run/prediction_manifest.json", "reports/batch_prediction_report.json")}
        manifest = {"files": files, "safety_flags": {"current_labels_accessed": False, "automatic_enforcement": False}, "model_included": False}
        self._write(root, "deployment_manifest.json", json.dumps(manifest)); return root

    def test_valid_candidate_only_bundle_has_stable_hashes(self):
        root = self._bundle(); first = validate_bundle(root); second = validate_bundle(root)
        self.assertEqual(first["files"], second["files"])

    def test_target_or_outcome_field_is_rejected(self):
        root = self._bundle(); ranked = root / "artifacts/predictions/run/ranked_candidates.csv"
        ranked.write_text("rank,activity_nr,raw_risk_score,serious_violation_found\n1,a,0.4,1\n", encoding="utf-8")
        with self.assertRaises(DeploymentBundleError): validate_bundle(root)

    def test_corruption_and_prohibited_runtime_state_are_rejected(self):
        root = self._bundle(); (root / "artifacts/predictions/run/ranked_candidates.csv").write_text("corrupt", encoding="utf-8")
        with self.assertRaises(DeploymentBundleError): validate_bundle(root)

    def test_runtime_dependencies_and_docker_contract_are_minimal(self):
        runtime = Path("requirements-runtime.txt").read_text(encoding="utf-8").lower()
        docker = Path("Dockerfile.deploy").read_text(encoding="utf-8")
        self.assertNotIn("mlflow", runtime)
        self.assertIn("USER appuser", docker)
        self.assertIn("${PORT:-8501}", docker)
        self.assertNotIn("curl", docker.lower())


if __name__ == "__main__": unittest.main()
