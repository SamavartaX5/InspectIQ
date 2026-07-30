import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_baseline import run_baseline
from src.baseline import build_industry_rates, choose_naics_grouping, score_validation_rows
from src.data_foundation import sha256_file, write_csv
from src.ranking_metrics import average_precision, ranking_at_fraction, roc_auc, selection_count
from src.splitting import SplitError, create_chronological_split, write_split_artifacts


def row(identifier, date, label=0, naics="111111"):
    return {
        "activity_nr": str(identifier), "open_date": date, "site_state": "CA", "naics_code": naics,
        "sic_code": "", "insp_type": "C", "insp_scope": "P", "owner_type": "A", "safety_hlth": "S",
        "nr_in_estab": "10", "serious_violation_found": label,
    }


class ChronologicalSplitTests(unittest.TestCase):
    def test_preferred_split_is_nonempty_disjoint_and_strictly_chronological(self):
        rows = [row("20", "2020-01-02"), row("21", "2021-01-02"), row("22", "2022-01-02"), row("23", "2023-01-02"), row("24", "2024-01-02")]
        split = create_chronological_split(rows)
        self.assertEqual(split["strategy"], "preferred_2020_2022_train_2023_validation_2024_test")
        self.assertEqual(split["periods"]["train"]["row_count"], 3)
        self.assertTrue(split["strictly_ordered"])
        self.assertEqual(split["id_overlap_count"], 0)

    def test_fallback_uses_three_available_labelled_periods(self):
        rows = [row("20", "2020-01-02"), row("21", "2021-01-02"), row("22", "2022-01-02"), row("23", "2023-01-02")]
        split = create_chronological_split(rows)
        self.assertEqual(split["selected_years"], {"train": [2020, 2021], "validation": [2022], "test": [2023]})

    def test_less_than_three_years_fails(self):
        with self.assertRaises(SplitError):
            create_chronological_split([row("20", "2020-01-02"), row("21", "2021-01-02")])

    def test_locked_test_file_has_no_label_column(self):
        split = create_chronological_split([row("20", "2020-01-02"), row("21", "2021-01-02"), row("22", "2022-01-02")])
        with tempfile.TemporaryDirectory() as directory:
            write_split_artifacts(Path(directory), split)
            header = (Path(directory) / "test_locked.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertNotIn("serious_violation_found", header)


class BaselineTests(unittest.TestCase):
    def training_rows(self):
        return [row(f"p{index}", "2020-01-02", 1, "111111") for index in range(20)] + [row(f"n{index}", "2020-02-02", 0, "222222") for index in range(20)]

    def test_training_only_rates_and_smoothing(self):
        rates = build_industry_rates(self.training_rows(), 3, alpha=5.0)
        self.assertEqual(rates["global_positive_rate"], 0.5)
        self.assertAlmostEqual(rates["groups"]["111"]["smoothed_rate"], 0.9)
        validation = [row("v", "2022-01-02", 0, "111999")]
        self.assertAlmostEqual(score_validation_rows(validation, rates)[0]["baseline_score"], 0.9)

    def test_validation_rows_are_not_mutated_or_used_for_rates(self):
        validation = [row("v", "2022-01-02", 1, "111999")]
        original = copy.deepcopy(validation)
        rates = build_industry_rates(self.training_rows(), 3)
        score_validation_rows(validation, rates)
        self.assertEqual(validation, original)
        changed_validation = [row("v", "2022-01-02", 0, "111999")]
        self.assertEqual(score_validation_rows(changed_validation, rates)[0]["baseline_score"], score_validation_rows(validation, rates)[0]["baseline_score"])

    def test_all_fallback_types_are_explicit(self):
        rates = build_industry_rates(self.training_rows(), 3)
        validation = [
            row("1", "2022-01-02", 1, "111999"), row("2", "2022-01-02", 0, "222999"),
            row("3", "2022-01-02", 0, "333999"), row("4", "2022-01-02", 0, ""),
        ]
        sources = {item["activity_nr"]: item["score_source"] for item in score_validation_rows(validation, rates, minimum_group_size=20)}
        self.assertEqual(sources, {"1": "industry_rate", "2": "industry_rate", "3": "unseen_group_fallback", "4": "missing_industry_fallback"})
        sparse_rates = build_industry_rates([row("a", "2020-01-02", 1, "444444"), row("b", "2020-01-02", 0, "111111")], 3)
        self.assertEqual(score_validation_rows([row("s", "2022-01-02", 0, "444999")], sparse_rates, minimum_group_size=2)[0]["score_source"], "sparse_group_fallback")

    def test_deterministic_tie_order_and_measured_group_choice(self):
        rates = build_industry_rates(self.training_rows(), 3)
        scored = score_validation_rows([row("10", "2022-01-02", 0, "333333"), row("2", "2022-01-02", 0, "444444")], rates)
        self.assertEqual([item["activity_nr"] for item in scored], ["2", "10"])
        digits, detail = choose_naics_grouping(self.training_rows())
        self.assertIn(digits, (2, 3))
        self.assertIn("adequate_row_coverage", detail["candidates"][digits])

    def test_ranking_metrics_use_ceil_and_handle_missing_class(self):
        rows = [{"baseline_score": 1 - index / 20, "actual_label": 1 if index in (0, 3, 7) else 0} for index in range(11)]
        self.assertEqual(selection_count(11, 0.05), 1)
        self.assertEqual(ranking_at_fraction(rows, 0.20)["selected"], 3)
        self.assertGreater(average_precision(rows), 0)
        self.assertIsNotNone(roc_auc(rows))
        self.assertIsNone(roc_auc([{"baseline_score": 1, "actual_label": 1}]))

    def test_offline_smoke_never_reports_locked_test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data" / "processed" / "snapshot"
            rows = [
                row("1", "2020-01-02", 1), row("2", "2020-02-02", 0),
                row("3", "2021-01-02", 1), row("4", "2021-02-02", 0),
                row("5", "2022-01-02", 1), row("6", "2022-02-02", 0),
                row("7", "2023-01-02", 1), row("8", "2023-02-02", 0),
            ]
            labelled = processed / "labelled_inspections.csv"
            write_csv(labelled, rows, list(rows[0]))
            report_path = root / "reports" / "data_foundation_report.json"
            report_path.parent.mkdir()
            report_path.write_text(json.dumps({"snapshot_id": "fixture", "processed_output": {"path": "data\\processed\\snapshot", "hashes": {"labelled_inspections.csv": sha256_file(labelled)}}}), encoding="utf-8")
            with patch("scripts.dol_api.DOLApiClient.get_records", side_effect=AssertionError("network request attempted")):
                report = run_baseline(
                    foundation_report_path=report_path, schema_path=Path("config/schema.yaml"),
                    artifact_root=root / "baseline",
                )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["baseline"]["training_only"])
        self.assertFalse(report["locked_test"]["metrics_calculated"])
        self.assertNotIn("metrics", report["locked_test"])


if __name__ == "__main__":
    unittest.main()
