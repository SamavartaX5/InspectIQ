import tempfile
import unittest
from pathlib import Path

from src.data_foundation import (
    LABEL_MAPPING_VERSION,
    build_processed_tables,
    create_raw_snapshot,
    run_foundation,
    stable_snapshot_id,
    verify_immutable,
)
from src.validation import ValidationError, load_schema, validate_inspections, validate_output_columns


class DataFoundationTests(unittest.TestCase):
    def schema(self):
        return {
            "schema_version": "test-v1",
            "inspection_fields": [
                {"name": "activity_nr", "required": True, "non_null": True, "feature": True},
                {"name": "open_date", "required": True, "non_null": True, "feature": True},
                {"name": "site_state", "required": True, "non_null": True, "feature": True},
                {"name": "naics_code", "required": True, "feature": True},
                {"name": "sic_code", "required": True, "feature": True},
                {"name": "insp_type", "required": True, "feature": True},
            ],
            "prohibited_leakage_columns": ["viol_type", "citation_id", "delete_flag", "current_penalty"],
        }

    def inspection(self, activity, date="2020-01-02T00:00:00"):
        return {"activity_nr": activity, "open_date": date, "site_state": "CA", "naics_code": "111111", "sic_code": None, "insp_type": "C"}

    def test_snapshot_id_is_deterministic(self):
        config = {"state": "CA", "start_date": "2020-01-01", "end_date": "2024-12-31"}
        first = stable_snapshot_id(config, ["2", "1"], ["2"])
        self.assertEqual(first, stable_snapshot_id(config, ["1", "2"], ["2"]))

    def test_immutable_snapshot_reuse_and_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"state": "CA", "start_date": "2020-01-01", "end_date": "2020-12-31"}
            identifier = stable_snapshot_id(config, ["1"], ["1"])
            args = dict(snapshot_root=root, snapshot_id=identifier, configuration=config, inspections=[self.inspection(1)], violations=[{"activity_nr": 1, "citation_id": "1", "viol_type": "S", "delete_flag": None}], completed_ids={"1"}, sources=["cache/manifest.json"])
            directory_one, _manifest, reused = create_raw_snapshot(**args)
            self.assertFalse(reused)
            _directory_two, _manifest_two, reused = create_raw_snapshot(**args)
            self.assertTrue(reused)
            (directory_one / "inspections.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaises(Exception):
                verify_immutable(directory_one, "manifest.json")

    def test_validation_detects_duplicate_invalid_date_and_required_column(self):
        rows = [self.inspection(1), self.inspection(1), {"activity_nr": 2, "open_date": "bad", "site_state": "CA"}]
        accepted, rejected, metrics = validate_inspections(rows, self.schema(), state="CA", start_date="2020-01-01", end_date="2020-12-31")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(metrics["duplicate_inspection_count"], 1)
        self.assertEqual(metrics["invalid_date_count"], 1)

    def test_join_labels_deleted_unknown_and_incomplete_are_safe(self):
        inspections = [self.inspection(index) for index in range(1, 7)]
        violations = [
            {"activity_nr": 1, "citation_id": "1", "viol_type": "S", "delete_flag": None},
            {"activity_nr": 2, "citation_id": "1", "viol_type": "O", "delete_flag": None},
            {"activity_nr": 4, "citation_id": "1", "viol_type": "Z", "delete_flag": None},
            {"activity_nr": 5, "citation_id": "1", "viol_type": "S", "delete_flag": "X"},
        ]
        labelled, excluded, rejected, _metrics = build_processed_tables(inspections, violations, {str(index) for index in range(1, 6)}, self.schema(), {"state": "CA", "start_date": "2020-01-01", "end_date": "2020-12-31"})
        labels = {str(row["activity_nr"]): row["serious_violation_found"] for row in labelled}
        self.assertEqual(labels, {"1": 1, "2": 0, "3": 0, "5": 0})
        self.assertEqual({str(row["activity_nr"]) for row in excluded}, {"4", "6"})
        self.assertEqual(rejected, [])

    def test_output_contract_excludes_leakage_columns(self):
        validate_output_columns(["activity_nr", "open_date", "site_state", "naics_code", "sic_code", "insp_type", "serious_violation_found"], self.schema())
        with self.assertRaises(ValidationError):
            validate_output_columns(["activity_nr", "open_date", "site_state", "naics_code", "sic_code", "insp_type", "serious_violation_found", "viol_type"], self.schema())

    def test_end_to_end_offline_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_foundation(
                report_path=Path("reports/feasibility_report.json"), schema_path=Path("config/schema.yaml"),
                snapshot_root=root / "raw", processed_root=root / "processed",
            )
        self.assertEqual(report["label_counts"]["labelled"], 2100)
        self.assertEqual(report["label_counts"]["positive"], 552)
        self.assertEqual(report["label_counts"]["negative"], 1548)
        self.assertEqual(report["excluded_table_shape"][0], 900)


if __name__ == "__main__":
    unittest.main()
