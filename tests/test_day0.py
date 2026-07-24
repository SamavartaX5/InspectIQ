import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from run_feasibility import (
    build_label_table,
    cache_key,
    cached_pages,
    chronological_split_summary,
    duplicate_activity_id_count,
    feasibility_checks,
    assess,
    request_definition,
    retrieve_violations,
    retrieve_balanced_inspections,
    run,
    write_json,
    yearly_row_budgets,
)
from scripts.dol_api import DOLApiClient, DOLApiError


class DOLApiClientTests(unittest.TestCase):
    def client(self, responses):
        session = Mock()
        session.get.side_effect = responses
        with patch("scripts.dol_api.load_dotenv"), patch("scripts.dol_api.os.getenv", return_value="secret"):
            return DOLApiClient(session=session, logger=lambda _: None), session

    def response(self, status, payload=None, headers=None):
        result = Mock(spec=requests.Response)
        result.status_code = status
        result.headers = headers or {}
        result.content = b"" if payload is None else b"{}"
        result.json.return_value = payload
        return result

    def test_204_is_valid_empty_result(self):
        client, _ = self.client([self.response(204)])
        self.assertEqual(client.get_records("violation"), [])

    @patch("scripts.dol_api.time.sleep")
    def test_429_retries_never_becomes_empty(self, _sleep):
        client, session = self.client([self.response(429, headers={"Retry-After": "0"}), self.response(200, {"data": [{"activity_nr": 1}]})])
        self.assertEqual(client.get_records("violation"), [{"activity_nr": 1}])
        self.assertEqual(session.get.call_count, 2)


class LabelTests(unittest.TestCase):
    def test_unknown_type_is_not_a_negative(self):
        labels, details = build_label_table([{"activity_nr": 1}], [{"activity_nr": 1, "viol_type": "Z", "delete_flag": None}], {"1"})
        self.assertIsNone(labels[0]["label"])
        self.assertEqual(details["unknown_violation_category_counts"], {"Z": 1})

    def test_duplicate_inspection_ids_are_detected(self):
        self.assertEqual(duplicate_activity_id_count([{"activity_nr": 1}, {"activity_nr": 1}]), 1)

    def test_incomplete_violation_batch_remains_unknown(self):
        labels, _ = build_label_table([{"activity_nr": 1}], [], set())
        self.assertIsNone(labels[0]["label"])
        self.assertEqual(labels[0]["label_exclusion_reason"], "violation_retrieval_incomplete")

    def test_429_path_cannot_become_a_negative_label(self):
        # A failed batch contributes no completed activity IDs.
        labels, _ = build_label_table([{"activity_nr": 1}], [], set())
        self.assertNotEqual(labels[0]["label"], 0)


class CacheResumeTests(unittest.TestCase):
    def cache_page(self, root, request, rows, *, complete=True):
        write_json(root / "page_00000000.json", rows)
        write_json(root / "manifest.json", {
            "request": request,
            "pages": {"0": {"offset": 0, "file": "page_00000000.json", "status": "success", "row_count": len(rows)}},
            "complete": complete,
        })

    def test_complete_inspection_cache_makes_zero_http_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ["activity_nr", "open_date"]
            filters = {"field": "site_state", "operator": "eq", "value": "CA"}
            request = request_definition("inspection", fields, filters, sort_by="open_date", sort="asc", wanted_rows=1)
            self.cache_page(root, request, [{"activity_nr": 1, "open_date": "2020-01-01"}])
            client = Mock()
            rows, complete, hit, error = cached_pages(client, cache_dir=root, endpoint="inspection", fields=fields, filter_object=filters, sort_by="open_date", sort="asc", wanted_rows=1)
            self.assertEqual(rows, [{"activity_nr": 1, "open_date": "2020-01-01"}])
            self.assertTrue(complete)
            self.assertTrue(hit)
            self.assertIsNone(error)
            client.get_records.assert_not_called()

    def test_offline_mode_makes_zero_http_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ["activity_nr"]
            filters = {"field": "site_state", "operator": "eq", "value": "CA"}
            request = request_definition("inspection", fields, filters, wanted_rows=1)
            write_json(root / "manifest.json", {"request": request, "pages": {}, "complete": False})
            client = Mock()
            rows, complete, hit, error = cached_pages(client, cache_dir=root, endpoint="inspection", fields=fields, filter_object=filters, wanted_rows=1, offline=True)
            self.assertEqual(rows, [])
            self.assertFalse(complete)
            self.assertFalse(hit)
            self.assertIsNone(error)
            client.get_records.assert_not_called()

    def test_completed_violation_ids_are_reused_across_compatible_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["1"]
            filters = {"field": "activity_nr", "operator": "in", "value": ids}
            batch = root / "audit_old" / "violation" / f"batch_0001_{cache_key(filters)}"
            request = request_definition("violation", ["activity_nr", "citation_id", "viol_type", "delete_flag"], filters)
            self.cache_page(batch, request, [{"activity_nr": 1, "citation_id": "1", "viol_type": "O", "delete_flag": None}])
            client = Mock()
            with patch("run_feasibility.CACHE_ROOT", root):
                rows, completed_ids, complete, error = retrieve_violations(client, ids, cache_base=root / "audit_new" / "violation", refresh=False, offline=False, max_new_batches=1, request_pause_seconds=0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(completed_ids, {"1"})
            self.assertTrue(complete)
            self.assertIsNone(error)
            client.get_records.assert_not_called()

    def test_transient_error_preserves_existing_detailed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "feasibility_report.json"
            attempt_path = Path(directory) / "feasibility_attempt_error.json"
            report_path.write_text('{"decision":"NO_GO","shapes":{"inspection":[3000,10]}}', encoding="utf-8")
            args = argparse.Namespace(state="CA", start_date="2020-01-01", end_date="2024-12-31", row_limit=3000, refresh=False, offline=False, max_new_violation_batches=1, request_pause_seconds=12.0)
            with patch("run_feasibility.REPORT_PATH", report_path), patch("run_feasibility.ATTEMPT_ERROR_PATH", attempt_path), patch("run_feasibility.build_report", side_effect=DOLApiError("temporary 429")):
                self.assertIsNone(run(args))
            self.assertEqual(report_path.read_text(encoding="utf-8"), '{"decision":"NO_GO","shapes":{"inspection":[3000,10]}}')
            self.assertIn("temporary 429", attempt_path.read_text(encoding="utf-8"))


class ChronologyTests(unittest.TestCase):
    @staticmethod
    def rows_for_year(year, count, prefix):
        return [
            {"activity_nr": f"{prefix}-{index}", "open_date": f"{year}-06-{(index % 28) + 1:02d}", "site_state": "CA"}
            for index in range(count)
        ]

    def test_default_sample_spans_each_requested_year_when_available(self):
        configuration = {"state": "CA", "start_date": "2020-01-01", "end_date": "2024-12-31", "row_limit": 3000}
        def fake_cached_pages(*_args, **kwargs):
            year = int(kwargs["cache_dir"].name.split("_")[1])
            return self.rows_for_year(year, 600, str(year)), True, False, None
        with tempfile.TemporaryDirectory() as directory, patch("run_feasibility.cached_inspection_records", return_value={}), patch("run_feasibility.cached_pages", side_effect=fake_cached_pages):
            rows, complete, _hit, error, coverage = retrieve_balanced_inspections(None, configuration, cache_root=Path(directory), refresh=False, offline=False)
        self.assertTrue(complete)
        self.assertIsNone(error)
        self.assertEqual({row["open_date"][:4] for row in rows}, {"2020", "2021", "2022", "2023", "2024"})
        self.assertEqual({year: values["requested_rows"] for year, values in coverage.items()}, {str(year): 600 for year in range(2020, 2025)})

    def test_row_allocation_is_balanced_and_ids_are_unique(self):
        budgets = yearly_row_budgets("2020-01-01", "2024-12-31", 3002)
        self.assertLessEqual(max(budgets.values()) - min(budgets.values()), 1)
        configuration = {"state": "CA", "start_date": "2020-01-01", "end_date": "2024-12-31", "row_limit": 5}
        def duplicate_cached_pages(*_args, **kwargs):
            year = int(kwargs["cache_dir"].name.split("_")[1])
            return [{"activity_nr": "shared", "open_date": f"{year}-06-01", "site_state": "CA"}], True, False, None
        with tempfile.TemporaryDirectory() as directory, patch("run_feasibility.cached_inspection_records", return_value={}), patch("run_feasibility.cached_pages", side_effect=duplicate_cached_pages):
            rows, _complete, _hit, _error, _coverage = retrieve_balanced_inspections(None, configuration, cache_root=Path(directory), refresh=False, offline=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(duplicate_activity_id_count(rows), 0)

    def test_chronological_split_periods_do_not_overlap(self):
        rows = []
        for year in (2020, 2021, 2022, 2023, 2024):
            rows.extend(self.rows_for_year(year, 1, str(year)))
        split = chronological_split_summary(rows)
        self.assertTrue(split["possible"])
        periods = split["periods"]
        self.assertLess(periods["train"]["max_open_date"], periods["validation"]["min_open_date"])
        self.assertLess(periods["validation"]["max_open_date"], periods["test"]["min_open_date"])


class FeasibilityDecisionTests(unittest.TestCase):
    def checks(self, *, labelled_count=2100, both_classes=True, chronological=True):
        return feasibility_checks(
            labelled_count=labelled_count,
            reliable_identifier=True,
            usable_open_date=True,
            confirmed_label_mapping=True,
            unknown_types_safe=True,
            has_both_classes=both_classes,
            candidate_features_supported=True,
            labelled_chronological_split_possible=chronological,
            industry_baseline_possible=True,
        )

    def test_2100_labelled_rows_pass_row_count_and_partial_retrieval_is_not_a_go_blocker(self):
        checks = self.checks(labelled_count=2100)
        self.assertTrue(checks["minimum_usable_labelled_inspections"])
        decision, failures, _ = assess({"feasibility_checks": checks})
        self.assertEqual(decision, "GO")
        self.assertEqual(failures, [])

    def test_fewer_than_2000_labelled_rows_fails(self):
        checks = self.checks(labelled_count=1999)
        self.assertFalse(checks["minimum_usable_labelled_inspections"])
        decision, failures, _ = assess({"feasibility_checks": checks})
        self.assertEqual(decision, "NO_GO")
        self.assertIn("minimum_usable_labelled_inspections", failures)

    def test_labelled_subset_requires_chronological_split(self):
        checks = self.checks(chronological=False)
        self.assertFalse(checks["labelled_subset_chronological_split_possible"])
        self.assertEqual(assess({"feasibility_checks": checks})[0], "NO_GO")

    def test_positive_and_negative_examples_are_required(self):
        checks = self.checks(both_classes=False)
        self.assertFalse(checks["positive_and_negative_examples"])
        self.assertEqual(assess({"feasibility_checks": checks})[0], "NO_GO")


if __name__ == "__main__":
    unittest.main()
