import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_feature_engineering as feature_runner
from run_feature_engineering import run_feature_engineering
from src.feature_engineering import (
    FEATURE_OUTPUT_FILES,
    FeatureEngineeringError,
    PROHIBITED_FEATURE_NAMES,
    TARGET_COLUMN,
    build_training_features,
    feature_columns,
    inspect_establishment_key,
    load_feature_config,
    normalized_establishment_composite,
    transform_batch,
    write_feature_artifacts,
)


CONFIG = {
    "feature_version": "test-v1", "industry_naics_digits": 2, "historical_rate_alpha": 5.0,
    "static_features": [], "employee_count": {"minimum_coverage_percentage": 95.0, "require_non_negative_integer": True},
}


def row(identifier, date, label=None, naics="111111"):
    value = {
        "activity_nr": str(identifier), "open_date": date, "naics_code": naics, "insp_type": "A", "insp_scope": "B",
        "owner_type": "C", "safety_hlth": "S", "nr_in_estab": "10", "site_state": "CA", "sic_code": "",
    }
    if label is not None:
        value[TARGET_COLUMN] = label
    return value


class HistoricalFeatureTests(unittest.TestCase):
    def artifact_fixture(self):
        outputs = {
            "train_features.csv": ([{"activity_nr": "1", TARGET_COLUMN: 0}], ["activity_nr", TARGET_COLUMN]),
            "validation_features.csv": ([{"activity_nr": "2", TARGET_COLUMN: 1}], ["activity_nr", TARGET_COLUMN]),
            "test_locked_features.csv": ([{"activity_nr": "3"}], ["activity_nr"]),
        }
        manifest = {
            "source_snapshot_id": "test", "feature_version": "test-v1",
            "output_row_counts": {"train": 1, "validation": 1, "test_locked": 1},
            "output_columns": {"train": ["activity_nr", TARGET_COLUMN], "validation": ["activity_nr", TARGET_COLUMN], "test_locked": ["activity_nr"]},
        }
        return outputs, manifest

    def test_current_target_cannot_affect_its_own_features_and_cold_start_is_explicit(self):
        features = build_training_features([row("1", "2020-01-01", 1)], CONFIG)
        self.assertEqual(features[0]["industry_prior_inspection_count"], 0)
        self.assertEqual(features[0]["industry_prior_positive_count"], 0)
        self.assertEqual(features[0]["industry_prior_positive_rate_smoothed"], 0.0)
        self.assertEqual(features[0]["industry_history_status"], "cold_start_global_fallback")

    def test_future_label_cannot_change_earlier_training_feature(self):
        original = [row("1", "2020-01-01", 1), row("2", "2020-01-02", 0), row("3", "2020-01-03", 0)]
        altered = copy.deepcopy(original)
        altered[2][TARGET_COLUMN] = 1
        self.assertEqual(build_training_features(original, CONFIG)[:2], build_training_features(altered, CONFIG)[:2])

    def test_same_day_rows_cannot_affect_each_other_even_with_different_times(self):
        features = build_training_features([row("2", "2020-01-01T12:00:00", 1), row("1", "2020-01-01T00:00:00", 0)], CONFIG)
        self.assertEqual([item["activity_nr"] for item in features], ["1", "2"])
        self.assertEqual([item["industry_prior_inspection_count"] for item in features], [0, 0])
        self.assertEqual([item["industry_prior_positive_count"] for item in features], [0, 0])

    def test_validation_labels_cannot_affect_validation_features(self):
        history = [row("h", "2020-01-01", 1)]
        candidates = [row("v1", "2021-01-01", 0), row("v2", "2021-01-01", 1)]
        changed = copy.deepcopy(candidates)
        changed[0][TARGET_COLUMN], changed[1][TARGET_COLUMN] = 1, 0
        self.assertEqual(transform_batch(history, candidates, CONFIG), transform_batch(history, changed, CONFIG))

    def test_test_labels_are_not_loaded_or_used(self):
        history = [row("h", "2020-01-01", 1)]
        candidate_with_label = [row("t", "2021-01-01", 0)]
        candidate_without_label = [row("t", "2021-01-01")]
        with_label = transform_batch(history, candidate_with_label, CONFIG)
        without_label = transform_batch(history, candidate_without_label, CONFIG)
        self.assertEqual(with_label, without_label)
        self.assertNotIn(TARGET_COLUMN, with_label[0])

    def test_validation_uses_training_history_only(self):
        train = [row("t", "2020-01-01", 1)]
        validation = [row("v1", "2021-01-01", 0), row("v2", "2021-01-02", 1)]
        output = transform_batch(train, validation, CONFIG)
        self.assertEqual([item["industry_prior_inspection_count"] for item in output], [1, 1])
        self.assertEqual([item["industry_prior_positive_count"] for item in output], [1, 1])

    def test_test_uses_training_plus_validation_history_only(self):
        train = [row("t", "2020-01-01", 1)]
        validation = [row("v", "2021-01-01", 0)]
        test = [row("x", "2022-01-01")]
        output = transform_batch([*train, *validation], test, CONFIG)
        self.assertEqual(output[0]["industry_prior_inspection_count"], 2)
        self.assertEqual(output[0]["industry_prior_positive_count"], 1)

    def test_prior_counts_and_smoothed_rate_are_correct(self):
        history = [row("a", "2020-01-01", 1, "111111"), row("b", "2020-01-02", 0, "222222")]
        output = transform_batch(history, [row("c", "2021-01-01", naics="111999")], CONFIG)[0]
        self.assertEqual(output["industry_prior_inspection_count"], 1)
        self.assertEqual(output["industry_prior_positive_count"], 1)
        self.assertAlmostEqual(output["industry_prior_positive_rate_smoothed"], (1 + 5 * 0.5) / 6)

    def test_unseen_and_missing_industry_use_global_history_fallback(self):
        history = [row("a", "2020-01-01", 1, "111111"), row("b", "2020-01-02", 0, "222222")]
        outputs = {item["activity_nr"]: item for item in transform_batch(history, [row("u", "2021-01-01", naics="333333"), row("m", "2021-01-01", naics="")], CONFIG)}
        unseen, missing = outputs["u"], outputs["m"]
        self.assertEqual(unseen["industry_history_status"], "unseen_industry_fallback")
        self.assertEqual(missing["industry_history_status"], "missing_industry_fallback")
        self.assertEqual(unseen["industry_prior_inspection_count"], 0)
        self.assertEqual(missing["industry_prior_inspection_count"], 0)
        self.assertEqual(unseen["industry_prior_positive_rate_smoothed"], 0.5)
        self.assertEqual(missing["industry_prior_positive_rate_smoothed"], 0.5)

    def test_exact_key_normalization_is_deterministic_but_no_key_is_selected_when_fields_absent(self):
        first = {"establishment_name": " Acme   Works ", "site_address": " 1 Main St ", "site_zip": "90210"}
        second = {"establishment_name": "acme works", "site_address": "1 main st", "site_zip": "90210"}
        self.assertEqual(normalized_establishment_composite(first), normalized_establishment_composite(second))
        assessment = inspect_establishment_key([row("1", "2020-01-01", 0)])
        self.assertEqual(assessment["selected_strategy"], "none")

    def test_establishment_days_feature_is_omitted_without_a_defensible_key(self):
        output = build_training_features([row("1", "2020-01-01", 0)], CONFIG)[0]
        self.assertNotIn("days_since_previous_inspection", output)

    def test_output_ids_remain_disjoint_and_locked_test_has_no_target_or_prohibited_columns(self):
        training_input = [row("1", "2020-01-01", 0), row("4", "2020-01-02", 1)]
        train = build_training_features(training_input, CONFIG)
        validation = transform_batch([row("1", "2020-01-01", 0)], [row("2", "2021-01-01", 1)], CONFIG, include_target=True)
        test = transform_batch([row("1", "2020-01-01", 0), row("2", "2021-01-01", 1)], [row("3", "2022-01-01")], CONFIG)
        self.assertFalse({item["activity_nr"] for item in train} & {item["activity_nr"] for item in validation})
        self.assertFalse({item["activity_nr"] for item in test} & {item["activity_nr"] for item in validation})
        self.assertEqual(len(train), len(training_input))
        self.assertEqual({item["activity_nr"] for item in train}, {item["activity_nr"] for item in training_input})
        self.assertNotIn(TARGET_COLUMN, test[0])
        self.assertFalse(set(test[0]) & PROHIBITED_FEATURE_NAMES)

    def test_deterministic_artifact_hashes_are_reused(self):
        outputs, manifest = self.artifact_fixture()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "features"
            first, reused_first = write_feature_artifacts(output, outputs, manifest)
            second, reused_second = write_feature_artifacts(output, outputs, manifest)
        self.assertFalse(reused_first)
        self.assertTrue(reused_second)
        self.assertEqual(first["file_hashes"], second["file_hashes"])

    def test_incomplete_existing_feature_directory_is_quarantined_and_rebuilt_safely(self):
        outputs, manifest = self.artifact_fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "features"
            output.mkdir()
            (output / "train_features.csv").write_text("partial", encoding="utf-8")
            rebuilt, reused = write_feature_artifacts(output, outputs, manifest)
            self.assertFalse(reused)
            self.assertEqual(set(rebuilt["file_hashes"]), set(FEATURE_OUTPUT_FILES.values()))
            self.assertTrue(all((output / name).exists() for name in {"feature_manifest.json", *FEATURE_OUTPUT_FILES.values()}))
            self.assertTrue(list(root.glob("features.quarantine-*")))
            self.assertFalse(list(root.glob("features.tmp-*")))
            self.assertNotIn(TARGET_COLUMN, (output / "test_locked_features.csv").read_text(encoding="utf-8").splitlines()[0])

    def test_invalid_manifest_and_hash_mismatch_are_not_reused(self):
        outputs, manifest = self.artifact_fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "features"
            first, reused = write_feature_artifacts(output, outputs, manifest)
            self.assertFalse(reused)
            (output / "feature_manifest.json").write_text("not json", encoding="utf-8")
            rebuilt_after_manifest, reused = write_feature_artifacts(output, outputs, manifest)
            self.assertFalse(reused)
            self.assertEqual(first["file_hashes"], rebuilt_after_manifest["file_hashes"])
            (output / "train_features.csv").write_text("tampered", encoding="utf-8")
            rebuilt_after_hash, reused = write_feature_artifacts(output, outputs, manifest)
            self.assertFalse(reused)
            self.assertEqual(first["file_hashes"], rebuilt_after_hash["file_hashes"])
            self.assertGreaterEqual(len(list(root.glob("features.quarantine-*"))), 2)

    def test_schema_and_row_count_mismatch_are_not_reused_even_when_hashes_are_updated(self):
        outputs, manifest = self.artifact_fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "features"
            write_feature_artifacts(output, outputs, manifest)
            train_path = output / "train_features.csv"
            train_path.write_text(f"activity_nr,{TARGET_COLUMN}\n1,0\n9,1\n", encoding="utf-8")
            stored = json.loads((output / "feature_manifest.json").read_text(encoding="utf-8"))
            stored["file_hashes"]["train_features.csv"] = hashlib.sha256(train_path.read_bytes()).hexdigest()
            (output / "feature_manifest.json").write_text(json.dumps(stored), encoding="utf-8")
            _rebuilt, reused = write_feature_artifacts(output, outputs, manifest)
            self.assertFalse(reused)
            self.assertTrue(list(root.glob("features.quarantine-*")))

    def test_failed_run_does_not_overwrite_a_valid_report(self):
        writes = []
        with patch.object(feature_runner, "run_feature_engineering", side_effect=FeatureEngineeringError("simulated build failure")), patch.object(feature_runner, "write_json_atomic", side_effect=lambda path, value: writes.append((path, value))):
            feature_runner.main()
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][0], Path("reports/feature_engineering_attempt_error.json"))
        self.assertEqual(writes[0][1]["status"], "FAIL")

    def test_no_fuzzy_matching_library_or_algorithm_is_introduced(self):
        source = Path("src/feature_engineering.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("fuzzywuzzy", source)
        self.assertNotIn("rapidfuzz", source)
        self.assertNotIn("difflib", source)

    def test_offline_end_to_end_smoke(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split_directory = root / "data" / "processed" / "snapshot" / "baseline" / "splits"
            from src.splitting import create_chronological_split, write_split_artifacts
            split = create_chronological_split([
                row("1", "2020-01-01", 0), row("2", "2020-02-01", 1),
                row("3", "2021-01-01", 0), row("4", "2021-02-01", 1),
                row("5", "2022-01-01", 0), row("6", "2022-02-01", 1),
            ])
            write_split_artifacts(split_directory, split)
            reports = root / "reports"; reports.mkdir()
            foundation = reports / "data_foundation_report.json"
            baseline = reports / "baseline_report.json"
            foundation.write_text(json.dumps({"snapshot_id": "fixture"}), encoding="utf-8")
            baseline.write_text(json.dumps({"source_snapshot_id": "fixture", "split_artifacts": {"directory": "data\\processed\\snapshot\\baseline\\splits"}}), encoding="utf-8")
            report = run_feature_engineering(
                foundation_report_path=foundation, baseline_report_path=baseline,
                config_path=Path("config/feature_config.yaml"), artifact_root=root / "features",
            )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["splits"]["train"]["output_row_count"], 2)
        self.assertFalse(report["splits"]["test_locked"]["target_present"])


if __name__ == "__main__":
    unittest.main()
