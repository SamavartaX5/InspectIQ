"""InspectIQ Day 0: reproducible OSHA inspection-label feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from scripts.dol_api import DOLApiClient, DOLApiError


CACHE_ROOT = Path("data/raw/day0_cache")
REPORT_PATH = Path("reports/feasibility_report.json")
ATTEMPT_ERROR_PATH = Path("reports/feasibility_attempt_error.json")
PAGE_SIZE = 500
VIOLATION_BATCH_SIZE = 100  # Live-probed successfully on 2026-07-24.
VIOLATION_BATCH_PAUSE_SECONDS = 12.0  # The live endpoint rate-limited a faster burst.
INSPECTION_SELECTION_STRATEGY = "year_balanced_v1"
MIN_USABLE_INSPECTIONS = 3_000
MINIMUM_USABLE_LABELLED_INSPECTIONS = 2_000
INSPECTION_FIELDS = [
    "activity_nr", "open_date", "site_state", "naics_code", "sic_code",
    "insp_type", "insp_scope", "owner_type", "safety_hlth", "nr_in_estab",
]
VIOLATION_FIELDS = ["activity_nr", "citation_id", "viol_type", "delete_flag"]
POSITIVE_TYPES = {"S", "W", "R"}
NON_POSITIVE_TYPES = {"O", "U"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def token(value: Any) -> str:
    if value is None:
        return "<NULL>"
    text = str(value).strip()
    return text if text else "<BLANK>"


def activity_id(row: dict[str, Any]) -> str | None:
    value = token(row.get("activity_nr"))
    return None if value in {"<NULL>", "<BLANK>"} else value


def duplicate_activity_id_count(rows: list[dict[str, Any]]) -> int:
    ids = [value for value in (activity_id(row) for row in rows) if value]
    return len(ids) - len(set(ids))


def count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(token(row.get(field)) for row in rows).most_common())


def cache_key(specification: dict[str, Any]) -> str:
    stable = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]


def normalized_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """The complete normalized identity of a Day 0 audit snapshot."""
    return {
        "state": args.state.strip().upper(),
        "start_date": datetime.fromisoformat(args.start_date).date().isoformat(),
        "end_date": datetime.fromisoformat(args.end_date).date().isoformat(),
        "row_limit": int(args.row_limit),
    }


def request_definition(
    endpoint: str,
    fields: list[str],
    filter_object: dict[str, Any],
    *,
    sort_by: str | None = None,
    sort: str | None = None,
    wanted_rows: int | None = None,
) -> dict[str, Any]:
    return {
        "endpoint": endpoint, "filters": filter_object, "fields": fields,
        "sort_by": sort_by, "sort": sort, "page_size": PAGE_SIZE,
        "wanted_rows": wanted_rows,
    }


def resolve_audit_cache(configuration: dict[str, Any]) -> Path:
    """Selection strategy is part of the cache identity; old caches stay intact."""
    canonical = CACHE_ROOT / f"audit_{cache_key({
        'cache_schema': 3, 'configuration': configuration,
        'inspection_selection_strategy': INSPECTION_SELECTION_STRATEGY,
        'inspection_fields': INSPECTION_FIELDS, 'sort_by': 'open_date', 'sort': 'asc',
    })}"
    return canonical


def load_cached_pages(
    cache_dir: Path, manifest: dict[str, Any], wanted_rows: int | None
) -> list[dict[str, Any]] | None:
    """Validate manifest entries, page files, and declared row counts."""
    pages = manifest.get("pages")
    if not isinstance(pages, dict):
        return None
    rows: list[dict[str, Any]] = []
    offsets = sorted((int(key) for key in pages if key.isdigit()))
    if offsets != list(range(0, len(offsets) * PAGE_SIZE, PAGE_SIZE)):
        return None
    for offset in offsets:
        page = pages[str(offset)]
        if not isinstance(page, dict) or page.get("status") != "success":
            return None
        file_name = page.get("file")
        if not isinstance(file_name, str):
            return None
        page_rows = read_json(cache_dir / file_name, None)
        if not isinstance(page_rows, list) or not all(isinstance(row, dict) for row in page_rows):
            return None
        if page.get("row_count") != len(page_rows):
            return None
        rows.extend(page_rows)
    if wanted_rows is not None and len(rows) < wanted_rows:
        return None
    return rows[:wanted_rows] if wanted_rows is not None else rows


def recover_inspection_snapshot(
    cache_dir: Path, request: dict[str, Any]
) -> dict[str, Any] | None:
    """Recover the pre-manifest Day 0 page files without trusting them blindly."""
    wanted_rows = request.get("wanted_rows")
    if not isinstance(wanted_rows, int) or wanted_rows < 1 or wanted_rows % PAGE_SIZE:
        return None
    page_files = [cache_dir / f"page_{offset:08d}.json" for offset in range(0, wanted_rows, PAGE_SIZE)]
    if not all(path.exists() for path in page_files):
        return None
    pages: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for offset, path in zip(range(0, wanted_rows, PAGE_SIZE), page_files):
        page_rows = read_json(path, None)
        if not isinstance(page_rows, list) or len(page_rows) != PAGE_SIZE or not all(isinstance(row, dict) for row in page_rows):
            return None
        pages[str(offset)] = {"offset": offset, "file": path.name, "status": "success", "row_count": len(page_rows), "recovered_at": now_iso()}
        all_rows.extend(page_rows)
    required = set(request["fields"])
    filter_state = request["filters"]["and"][0]["value"]
    start = request["filters"]["and"][1]["value"]
    end = request["filters"]["and"][2]["value"]
    dates = [str(row.get("open_date", "")) for row in all_rows]
    if (not all(required.issubset(row) for row in all_rows)
            or not all(str(row.get("site_state", "")).upper() == filter_state for row in all_rows)
            or dates != sorted(dates) or not all(start < value < end for value in dates)):
        return None
    return {"cache_schema": 2, "request": request, "pages": pages, "complete": True,
            "completion_reason": "recovered_requested_row_limit", "completed_at": now_iso()}


def cached_pages(
    client: DOLApiClient | None,
    *,
    cache_dir: Path,
    endpoint: str,
    fields: list[str],
    filter_object: dict[str, Any],
    sort_by: str | None = None,
    sort: str | None = None,
    wanted_rows: int | None = None,
    refresh: bool = False,
    offline: bool = False,
    announce: bool = True,
) -> tuple[list[dict[str, Any]], bool, bool, str | None]:
    """Fetch pages once, recording each successful page before proceeding."""
    manifest_path = cache_dir / "manifest.json"
    request = request_definition(endpoint, fields, filter_object, sort_by=sort_by, sort=sort, wanted_rows=wanted_rows)
    manifest = {} if refresh else read_json(manifest_path, {})
    if not refresh and manifest.get("request") == request:
        cached = load_cached_pages(cache_dir, manifest, wanted_rows)
        if cached is not None and manifest.get("complete"):
            if announce:
                print(f"CACHE HIT endpoint={endpoint} rows={len(cached)}")
            return cached, True, True, None
        if endpoint == "inspection":
            recovered = recover_inspection_snapshot(cache_dir, request)
            if recovered:
                write_json(manifest_path, recovered)
                cached = load_cached_pages(cache_dir, recovered, wanted_rows)
                if cached is not None:
                    if announce:
                        print(f"CACHE HIT endpoint=inspection rows={len(cached)} recovered_manifest=true")
                    return cached, True, True, None
    if manifest.get("request") != request:
        manifest = {"request": request, "pages": {}, "complete": False, "created_at": now_iso()}
    manifest.setdefault("pages", {})
    manifest["complete"] = bool(manifest.get("complete", False)) and not refresh
    if offline:
        cached = load_cached_pages(cache_dir, manifest, None) or []
        if announce:
            print(f"CACHE ONLY endpoint={endpoint} rows={len(cached)} complete={bool(manifest.get('complete'))}")
        return cached[:wanted_rows] if wanted_rows else cached, bool(manifest.get("complete")), False, None
    write_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page_key = str(offset)
        page = manifest["pages"].get(page_key, {})
        page_file = cache_dir / str(page.get("file", f"page_{offset:08d}.json"))
        if page.get("status") == "success" and page_file.exists() and not refresh:
            page_rows = read_json(page_file, [])
        elif manifest.get("complete"):
            break
        else:
            if client is None:
                raise DOLApiError("A DOL API client is required when offline mode is disabled.")
            try:
                page_rows = client.get_records(
                    endpoint, fields=fields, filter_object=filter_object,
                    sort_by=sort_by, sort=sort, limit=PAGE_SIZE, offset=offset,
                )
            except DOLApiError as error:
                manifest["pages"][page_key] = {
                    "offset": offset, "status": "failed", "retrieved_at": now_iso(),
                }
                write_json(manifest_path, manifest)
                if rows:
                    return rows[:wanted_rows] if wanted_rows else rows, False, False, str(error)
                raise
            file_name = f"page_{offset:08d}.json"
            write_json(cache_dir / file_name, page_rows)
            manifest["pages"][page_key] = {
                "offset": offset, "file": file_name, "status": "success",
                "retrieved_at": now_iso(), "row_count": len(page_rows),
            }
            write_json(manifest_path, manifest)

        if not isinstance(page_rows, list):
            raise DOLApiError("Cached page did not contain a record list.")
        rows.extend(row for row in page_rows if isinstance(row, dict))
        if wanted_rows is not None and len(rows) >= wanted_rows:
            manifest.update({"complete": True, "completion_reason": "requested_row_limit", "completed_at": now_iso()})
            write_json(manifest_path, manifest)
            return rows[:wanted_rows], True, False, None
        if len(page_rows) < PAGE_SIZE:
            manifest.update({"complete": True, "completion_reason": "end_of_results", "completed_at": now_iso()})
            write_json(manifest_path, manifest)
            return rows, True, False, None
        offset += PAGE_SIZE

    return rows[:wanted_rows] if wanted_rows else rows, bool(manifest.get("complete")), False, None


def date_filter(state: str, start_date: str, end_date: str) -> dict[str, Any]:
    start_inclusive = (datetime.fromisoformat(start_date).date() - timedelta(days=1)).isoformat()
    end_exclusive = (datetime.fromisoformat(end_date).date() + timedelta(days=1)).isoformat()
    return {"and": [
        {"field": "site_state", "operator": "eq", "value": state.upper()},
        # The verified API supports strict comparisons; using the prior day's
        # final second keeps the requested calendar start date inclusive.
        {"field": "open_date", "operator": "gt", "value": f"{start_inclusive}T23:59:59"},
        {"field": "open_date", "operator": "lt", "value": f"{end_exclusive}T00:00:00"},
    ]}


def yearly_row_budgets(start_date: str, end_date: str, row_limit: int) -> dict[int, int]:
    start_year = datetime.fromisoformat(start_date).year
    end_year = datetime.fromisoformat(end_date).year
    years = list(range(start_year, end_year + 1))
    base, remainder = divmod(row_limit, len(years))
    return {year: base + int(index < remainder) for index, year in enumerate(years)}


def year_window(year: int, start_date: str, end_date: str) -> tuple[str, str]:
    return max(start_date, f"{year}-01-01"), min(end_date, f"{year}-12-31")


def row_year(row: dict[str, Any]) -> int | None:
    value = str(row.get("open_date", ""))
    try:
        return datetime.fromisoformat(value).year
    except ValueError:
        return None


def cached_inspection_records() -> dict[str, dict[str, Any]]:
    """Index compatible historical pages by ID without modifying their caches."""
    records: dict[str, dict[str, Any]] = {}
    manifest_paths = set(CACHE_ROOT.glob("audit_*/inspection/manifest.json"))
    manifest_paths.update(CACHE_ROOT.glob("audit_*/inspection/year_*/manifest.json"))
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path, {})
        request = manifest.get("request", {})
        if request.get("endpoint") != "inspection" or not manifest.get("complete"):
            continue
        rows = load_cached_pages(manifest_path.parent, manifest, None)
        if rows is None:
            continue
        for row in rows:
            identifier = activity_id(row)
            if identifier:
                records.setdefault(identifier, row)
    return records


def materialize_reused_inspection_year(
    cache_dir: Path, request: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Create a new selection manifest using validated copies of cached rows."""
    pages: dict[str, Any] = {}
    for offset in range(0, len(rows), PAGE_SIZE):
        page_rows = rows[offset : offset + PAGE_SIZE]
        file_name = f"page_{offset:08d}.json"
        write_json(cache_dir / file_name, page_rows)
        pages[str(offset)] = {
            "offset": offset, "file": file_name, "status": "success",
            "row_count": len(page_rows), "source": "reused_cached_inspection_records",
        }
    write_json(cache_dir / "manifest.json", {
        "cache_schema": 3, "request": request, "pages": pages, "complete": True,
        "completion_reason": "reused_cached_records", "completed_at": now_iso(),
    })


def retrieve_balanced_inspections(
    client: DOLApiClient | None,
    configuration: dict[str, Any],
    *,
    cache_root: Path,
    refresh: bool,
    offline: bool,
) -> tuple[list[dict[str, Any]], bool, bool, str | None, dict[str, dict[str, int]]]:
    """Select each calendar year independently, never letting one year fill another's quota."""
    budgets = yearly_row_budgets(configuration["start_date"], configuration["end_date"], configuration["row_limit"])
    reusable = cached_inspection_records()
    all_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    complete = True
    cache_hit = True
    errors: list[str] = []
    coverage: dict[str, dict[str, int]] = {}
    selection_manifest = {
        "cache_schema": 3, "strategy": INSPECTION_SELECTION_STRATEGY,
        "configuration": configuration, "year_budgets": budgets, "years": {},
    }
    for year, budget in budgets.items():
        year_start, year_end = year_window(year, configuration["start_date"], configuration["end_date"])
        filter_object = date_filter(configuration["state"], year_start, year_end)
        request = request_definition("inspection", INSPECTION_FIELDS, filter_object, sort_by="open_date", sort="asc", wanted_rows=budget)
        year_cache = cache_root / "inspection" / f"year_{year}"
        existing_manifest = read_json(year_cache / "manifest.json", {})
        if not refresh and existing_manifest.get("request") != request:
            reusable_rows = [
                row for row in reusable.values()
                if row_year(row) == year and activity_id(row) not in seen_ids
            ]
            reusable_rows.sort(key=lambda row: (str(row.get("open_date", "")), activity_id(row) or ""))
            if len(reusable_rows) >= budget:
                materialize_reused_inspection_year(year_cache, request, reusable_rows[:budget])
        rows, year_complete, year_hit, year_error = cached_pages(
            client, cache_dir=year_cache, endpoint="inspection", fields=INSPECTION_FIELDS,
            filter_object=filter_object, sort_by="open_date", sort="asc", wanted_rows=budget,
            refresh=refresh, offline=offline,
        )
        retained = []
        for row in rows:
            identifier = activity_id(row)
            if identifier and identifier not in seen_ids:
                seen_ids.add(identifier)
                retained.append(row)
        all_rows.extend(retained)
        complete = complete and year_complete and len(retained) >= budget
        cache_hit = cache_hit and year_hit
        if year_error:
            errors.append(f"{year}: {year_error}")
        coverage[str(year)] = {
            "requested_rows": budget, "retrieved_rows": len(rows), "unique_rows_retained": len(retained),
        }
        selection_manifest["years"][str(year)] = {
            "request": request, "complete": year_complete, "cache_hit": year_hit,
            **coverage[str(year)], "error": year_error,
        }
    all_rows.sort(key=lambda row: (str(row.get("open_date", "")), activity_id(row) or ""))
    if not offline:
        selection_manifest["complete"] = complete
        selection_manifest["updated_at"] = now_iso()
        write_json(cache_root / "inspection" / "selection_manifest.json", selection_manifest)
    return all_rows, complete, cache_hit, "; ".join(errors) or None, coverage


def cached_violation_coverage(ids: set[str]) -> tuple[list[dict[str, Any]], set[str]]:
    """Reuse complete batches from any compatible Day 0 selection cache."""
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    seen_citations: set[tuple[str | None, str]] = set()
    for manifest_path in CACHE_ROOT.glob("audit_*/violation/batch_*/manifest.json"):
        manifest = read_json(manifest_path, {})
        request = manifest.get("request", {})
        filters = request.get("filters", {})
        batch_ids = {str(value) for value in filters.get("value", [])}
        if (request.get("endpoint") != "violation" or request.get("fields") != VIOLATION_FIELDS
                or not manifest.get("complete") or not batch_ids):
            continue
        cached = load_cached_pages(manifest_path.parent, manifest, None)
        if cached is None:
            continue
        matching = batch_ids & ids
        if not matching:
            continue
        completed.update(matching)
        for row in cached:
            identifier = activity_id(row)
            if identifier not in ids:
                continue
            key = (identifier, token(row.get("citation_id")))
            if key not in seen_citations:
                seen_citations.add(key)
                rows.append(row)
    return rows, completed


def retrieve_violations(
    client: DOLApiClient | None,
    ids: list[str],
    *,
    cache_base: Path,
    refresh: bool,
    offline: bool,
    max_new_batches: int,
    request_pause_seconds: float,
) -> tuple[list[dict[str, Any]], set[str], bool, str | None]:
    """Resume only IDs with no complete cached violation query."""
    wanted_ids = set(ids)
    all_rows, completed_ids = cached_violation_coverage(wanted_ids) if not refresh else ([], set())
    pending_ids = [identifier for identifier in ids if identifier not in completed_ids]
    overall = {
        "endpoint": "violation", "batch_size": VIOLATION_BATCH_SIZE,
        "reused_completed_activity_ids": len(completed_ids), "batches": [], "complete": False,
    }
    retrieval_error: str | None = None
    new_batches = 0
    for start in range(0, len(pending_ids), VIOLATION_BATCH_SIZE):
        if offline or new_batches >= max_new_batches:
            break
        batch = pending_ids[start : start + VIOLATION_BATCH_SIZE]
        filter_object = {"field": "activity_nr", "operator": "in", "value": batch}
        batch_dir = cache_base / f"batch_{cache_key(filter_object)}"
        try:
            rows, complete, cache_hit, batch_error = cached_pages(
                client, cache_dir=batch_dir, endpoint="violation", fields=VIOLATION_FIELDS,
                filter_object=filter_object, wanted_rows=None, refresh=refresh, offline=False,
                announce=False,
            )
        except DOLApiError as error:
            retrieval_error = str(error)
            overall["error"] = retrieval_error
            break
        if not cache_hit:
            new_batches += 1
        all_rows.extend(rows)
        if complete:
            completed_ids.update(batch)
        overall["batches"].append({
            "activity_id_count": len(batch), "row_count": len(rows),
            "complete": complete, "cache_hit": cache_hit,
        })
        if batch_error:
            retrieval_error = batch_error
            overall["error"] = retrieval_error
            break
        if start + VIOLATION_BATCH_SIZE < len(pending_ids) and not cache_hit:
            client._log("violation", "throttle", new_batches, 0, request_pause_seconds)
            time.sleep(request_pause_seconds)
    overall["complete"] = len(completed_ids) == len(wanted_ids)
    overall["updated_at"] = now_iso()
    if not overall["complete"] and retrieval_error is None:
        retrieval_error = (
            f"Violation retrieval is incomplete: {len(completed_ids)} of {len(wanted_ids)} "
            "inspection IDs have completed queries."
        )
    if offline:
        print(f"CACHE ONLY violation completed_ids={len(completed_ids)} total_ids={len(wanted_ids)} rows={len(all_rows)}")
    if not offline:
        write_json(cache_base / "manifest.json", overall)
    return all_rows, completed_ids, bool(overall["complete"]), retrieval_error


def build_label_table(
    inspections: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    retrieved_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Make one row per inspection and keep unknown categories out of negatives."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in violations:
        row_id = activity_id(row)
        if row_id:
            grouped[row_id].append(row)

    labels: list[dict[str, Any]] = []
    unknown_categories: Counter[str] = Counter()
    unknown_delete_flags: Counter[str] = Counter()
    excluded_deleted = 0
    for inspection in inspections:
        inspection_id = activity_id(inspection)
        label: int | None = None
        reason: str | None = None
        if not inspection_id or inspection_id not in retrieved_ids:
            reason = "violation_retrieval_incomplete"
        else:
            positive = False
            unknown = False
            for violation in grouped.get(inspection_id, []):
                delete_flag = token(violation.get("delete_flag"))
                if delete_flag == "X":
                    excluded_deleted += 1
                    continue
                if delete_flag not in {"<NULL>", "<BLANK>"}:
                    unknown_delete_flags[delete_flag] += 1
                    unknown = True
                    continue
                violation_type = token(violation.get("viol_type"))
                if violation_type in POSITIVE_TYPES:
                    positive = True
                elif violation_type not in NON_POSITIVE_TYPES:
                    unknown_categories[violation_type] += 1
                    unknown = True
            if positive:
                label = 1
            elif unknown:
                reason = "unknown_violation_category_or_delete_flag"
            else:
                label = 0
        labels.append({"activity_nr": inspection_id, "label": label, "label_exclusion_reason": reason})

    return labels, {
        "unknown_violation_category_counts": dict(unknown_categories.most_common()),
        "unknown_delete_flag_counts": dict(unknown_delete_flags.most_common()),
        "deleted_violation_rows_excluded": excluded_deleted,
    }


def missing_percentages(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, float]:
    total = len(rows)
    if not total:
        return {column: 100.0 for column in columns}
    return {
        column: round(100 * sum(token(row.get(column)) in {"<NULL>", "<BLANK>"} for row in rows) / total, 2)
        for column in columns
    }


def metadata_lookup(endpoint: str, field: str) -> dict[str, Any] | None:
    rows = read_json(Path("reports/schema") / f"{endpoint}_metadata.json", [])
    return next((row for row in rows if row.get("short_name") == field), None)


def catalog_bulk_assessment() -> dict[str, Any]:
    catalog = read_json(Path("reports/dol_osha_catalog.json"), [])
    relevant = [row for row in catalog if row.get("api_url") in {"inspection", "violation"}]
    bulk_keys = {"download_url", "download", "resources", "resource_url", "file_url", "url"}
    has_bulk = any(any(row.get(key) for key in bulk_keys) for row in relevant)
    return {
        "official_bulk_resource_found": has_bulk,
        "decision": "API used" if not has_bulk else "catalog bulk resource requires review",
        "reason": "The saved official catalog entries expose API endpoints but no current bulk CSV, ZIP, or download resource field.",
    }


def assess(report: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    checks = report["feasibility_checks"]
    failures = [name for name, passed in checks.items() if not passed]
    corrections = []
    if "minimum_usable_labelled_inspections" in failures:
        corrections.append("Resume violation retrieval until at least 2,000 inspections have completed labels.")
    if "confirmed_serious_willful_repeat_label_mapping" in failures:
        corrections.append("Refresh official violation metadata and confirm the label mapping before modelling.")
    if "positive_and_negative_examples" in failures:
        corrections.append("Expand the measured sample before considering a classifier.")
    if any(item in failures for item in {"reliable_activity_identifier", "usable_open_date"}):
        corrections.append("Investigate source-record quality; do not work around identifier or date defects in a model.")
    if "three_leakage_free_candidate_features" in failures:
        corrections.append("Establish an operational pre-ranking snapshot for three site attributes before advancing.")
    if "labelled_subset_chronological_split_possible" in failures:
        corrections.append("Collect labelled records in at least three non-overlapping chronological periods before modelling.")
    if "industry_or_rule_baseline_possible" in failures:
        corrections.append("Require usable NAICS/SIC coverage or use a documented non-industry rule baseline.")
    return ("GO" if not failures else "NO_GO"), failures, corrections


def feasibility_checks(
    *,
    labelled_count: int,
    reliable_identifier: bool,
    usable_open_date: bool,
    confirmed_label_mapping: bool,
    unknown_types_safe: bool,
    has_both_classes: bool,
    candidate_features_supported: bool,
    labelled_chronological_split_possible: bool,
    industry_baseline_possible: bool,
) -> dict[str, bool]:
    """GO criteria use only fully labelled records for label-dependent checks."""
    return {
        "minimum_usable_labelled_inspections": labelled_count >= MINIMUM_USABLE_LABELLED_INSPECTIONS,
        "reliable_activity_identifier": reliable_identifier,
        "usable_open_date": usable_open_date,
        "confirmed_serious_willful_repeat_label_mapping": confirmed_label_mapping,
        "unknown_violation_types_handled_safely": unknown_types_safe,
        "positive_and_negative_examples": has_both_classes,
        "three_leakage_free_candidate_features": candidate_features_supported,
        "labelled_subset_chronological_split_possible": labelled_chronological_split_possible,
        "industry_or_rule_baseline_possible": industry_baseline_possible,
    }


def chronological_split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_year: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        year = row_year(row)
        date_value = str(row.get("open_date", ""))
        if year is not None and date_value:
            by_year[year].append(date_value)
    years = sorted(by_year)
    if len(years) < 3:
        return {"possible": False, "represented_periods": len(years), "periods": {}}
    train_years, validation_years, test_years = years[:-2], [years[-2]], [years[-1]]
    period_years = {"train": train_years, "validation": validation_years, "test": test_years}
    periods = {
        name: {
            "years": selected_years,
            "row_count": sum(len(by_year[year]) for year in selected_years),
            "min_open_date": min(date for year in selected_years for date in by_year[year]),
            "max_open_date": max(date for year in selected_years for date in by_year[year]),
        }
        for name, selected_years in period_years.items()
    }
    ordered = [periods[name] for name in ("train", "validation", "test")]
    non_overlapping = all(
        ordered[index]["row_count"] > 0
        and ordered[index]["max_open_date"] < ordered[index + 1]["min_open_date"]
        for index in range(2)
    ) and ordered[-1]["row_count"] > 0
    return {"possible": non_overlapping, "represented_periods": len(years), "periods": periods}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    specification = normalized_configuration(args)
    root = resolve_audit_cache(specification)
    client = None if args.offline else DOLApiClient()
    inspections, inspection_complete, inspection_cache_hit, inspection_retrieval_error, year_coverage = retrieve_balanced_inspections(
        client, specification, cache_root=root, refresh=args.refresh, offline=args.offline,
    )
    ids = [value for value in (activity_id(row) for row in inspections) if value]
    unique_ids = list(dict.fromkeys(ids))
    violations, retrieved_ids, violations_complete, violation_retrieval_error = retrieve_violations(
        client, unique_ids, cache_base=root / "violation", refresh=args.refresh, offline=args.offline,
        max_new_batches=args.max_new_violation_batches, request_pause_seconds=args.request_pause_seconds,
    )
    label_rows, label_details = build_label_table(inspections, violations, retrieved_ids)
    labels_by_id = {row["activity_nr"]: row for row in label_rows}
    joined = [{**row, **labels_by_id.get(activity_id(row), {})} for row in inspections]
    labelled_rows = [row for row in joined if row.get("label") in {0, 1}]

    valid_dates = sorted(token(row.get("open_date")) for row in inspections if token(row.get("open_date")) not in {"<NULL>", "<BLANK>"})
    inspection_count_by_year = dict(Counter(str(row_year(row)) for row in inspections if row_year(row) is not None))
    duplicated_ids = duplicate_activity_id_count(inspections)
    positive = sum(row["label"] == 1 for row in label_rows)
    negative = sum(row["label"] == 0 for row in label_rows)
    excluded = sum(row["label"] is None for row in label_rows)
    naics_coverage = round(100 - missing_percentages(inspections, ["naics_code"])["naics_code"], 2)
    sic_coverage = round(100 - missing_percentages(inspections, ["sic_code"])["sic_code"], 2)
    industry_values = {token(row.get("naics_code")) for row in inspections} - {"<NULL>", "<BLANK>"}
    chronological_split = chronological_split_summary(inspections)
    labelled_chronological_split = chronological_split_summary(labelled_rows)
    label_counts_by_year: dict[str, dict[str, int]] = {}
    for row in labelled_rows:
        year = str(row_year(row))
        counts = label_counts_by_year.setdefault(year, {"positive": 0, "negative": 0, "labelled": 0})
        counts["labelled"] += 1
        counts["positive" if row["label"] == 1 else "negative"] += 1
    positive_metadata = metadata_lookup("violation", "viol_type") or {}
    delete_metadata = metadata_lookup("violation", "delete_flag") or {}
    allowed = str(positive_metadata.get("allowed_value_list", ""))
    mapping_confirmed = all(f"{code};" in allowed for code in POSITIVE_TYPES) and "O;Other" in allowed and "U;Unclassified" in allowed
    candidate_features = [
        {"field": "site_state", "reason": "site location, available by inspection open_date and not an outcome field", "coverage_percent": round(100 - missing_percentages(inspections, ["site_state"])["site_state"], 2)},
        {"field": "naics_code", "reason": "industry classification, available by inspection open_date and not an outcome field", "coverage_percent": naics_coverage},
        {"field": "owner_type", "reason": "establishment ownership classification, available by inspection open_date and not an outcome field", "coverage_percent": round(100 - missing_percentages(inspections, ["owner_type"])["owner_type"], 2)},
    ]
    candidates_supported = all(item["coverage_percent"] > 0 for item in candidate_features)
    unknown_types_safe = all(
        row["label"] is None
        for row in label_rows
        if row["label_exclusion_reason"] == "unknown_violation_category_or_delete_flag"
    )
    checks = feasibility_checks(
        labelled_count=positive + negative,
        reliable_identifier=bool(unique_ids) and len(unique_ids) == len(inspections) and duplicated_ids == 0,
        usable_open_date=len(valid_dates) == len(inspections),
        confirmed_label_mapping=mapping_confirmed,
        unknown_types_safe=unknown_types_safe,
        has_both_classes=positive > 0 and negative > 0,
        candidate_features_supported=candidates_supported and len(candidate_features) >= 3,
        labelled_chronological_split_possible=labelled_chronological_split["possible"],
        industry_baseline_possible=naics_coverage > 0 and len(industry_values) > 1,
    )
    retrieval_coverage_percentage = round(100 * len(retrieved_ids) / len(unique_ids), 2) if unique_ids else 0.0
    warnings = []
    if retrieval_coverage_percentage < 100:
        warnings.append({
            "type": "incomplete_violation_retrieval",
            "coverage_percentage": retrieval_coverage_percentage,
            "excluded_inspections": excluded,
            "message": "Metrics describe only inspections with completed violation retrieval; missing retrieval outcomes were not assumed negative.",
        })
    report: dict[str, Any] = {
        "audit": "InspectIQ Day 0 feasibility audit", "generated_at": now_iso(), "configuration": specification,
        "acquisition": {"method": "DOL OSHA API with cached, bounded activity_nr batches", "inspection_selection_strategy": INSPECTION_SELECTION_STRATEGY, "cache_directory": str(root), "offline": args.offline, "inspection_cache_hit": inspection_cache_hit, "inspection_complete": inspection_complete, "inspection_retrieval_error": inspection_retrieval_error, "violation_complete": violations_complete, "violation_retrieval_error": violation_retrieval_error, "catalog_bulk_assessment": catalog_bulk_assessment()},
        "label_definition": {"intended_positive": "at least one non-deleted Serious, Willful, or Repeat violation", "official_viol_type_mapping": {"S": "Serious", "W": "Willful", "R": "Repeat", "O": "Other", "U": "Unclassified"}, "metadata_allowed_value_list": positive_metadata.get("allowed_value_list"), "deleted_rows_excluded_when": "delete_flag is X (official metadata: X = Deleted)", "delete_flag_allowed_value_list": delete_metadata.get("allowed_value_list")},
        "shapes": {"inspection": [len(inspections), len({key for row in inspections for key in row})], "violation": [len(violations), len({key for row in violations for key in row})], "joined_inspection_level": [len(joined), len({key for row in joined for key in row})]},
        "columns": {"inspection": sorted({key for row in inspections for key in row}), "violation": sorted({key for row in violations for key in row}), "joined_inspection_level": sorted({key for row in joined for key in row})},
        "inspection_date_range": {"min": valid_dates[0] if valid_dates else None, "max": valid_dates[-1] if valid_dates else None},
        "inspection_year_coverage": year_coverage,
        "inspection_count_by_year": inspection_count_by_year,
        "represented_year_count": len({row_year(row) for row in inspections if row_year(row) is not None}),
        "chronological_split": chronological_split,
        "labelled_subset_chronological_split": labelled_chronological_split,
        "label_counts_by_year": label_counts_by_year,
        "state_coverage": count_values(inspections, "site_state"), "missing_value_percentages": missing_percentages(inspections, INSPECTION_FIELDS),
        "duplicate_activity_nr_count": duplicated_ids, "missing_activity_nr_count": len(inspections) - len(ids),
        "violation_retrieval_coverage": {"completed_inspection_ids": len(retrieved_ids), "total_inspection_ids": len(unique_ids), "percentage": retrieval_coverage_percentage},
        "inspection_to_violation_match_rate": {"matched_completed_inspection_ids": len({activity_id(row) for row in violations if activity_id(row)} & retrieved_ids), "completed_inspection_ids": len(retrieved_ids), "percentage_among_completed": round(100 * len({activity_id(row) for row in violations if activity_id(row)} & retrieved_ids) / len(retrieved_ids), 2) if retrieved_ids else 0.0},
        "label_coverage": {"labelled_inspections": positive + negative, "total_inspections": len(inspections), "percentage": round(100 * (positive + negative) / len(inspections), 2) if inspections else 0.0},
        "naics_coverage_percent": naics_coverage, "sic_coverage_percent": sic_coverage,
        "observed_viol_type_counts": count_values(violations, "viol_type"), "delete_flag_counts": count_values(violations, "delete_flag"),
        **label_details, "label_counts": {"positive": positive, "negative": negative, "excluded_or_unknown": excluded, "positive_class_percentage": round(100 * positive / (positive + negative), 2) if positive + negative else None},
        "candidate_leakage_free_preinspection_features": candidate_features,
        "likely_post_inspection_leakage_fields": ["viol_type", "citation_id", "delete_flag", "issuance_date", "initial_penalty", "current_penalty", "contest_date", "final_order_date", "case_mod_date", "close_case_date", "close_conf_date", "current-inspection violation counts", "future inspection records"],
        "minimum_usable_labelled_inspections": MINIMUM_USABLE_LABELLED_INSPECTIONS,
        "warnings": warnings,
        "scope_guardrails": ["Historical decision-support prototype only.", "Use only for supplied candidate ranking; never automated enforcement.", "No claim of real OSHA cost savings.", "Inspected establishments do not represent all workplaces."],
        "feasibility_checks": checks,
    }
    decision, failures, corrections = assess(report)
    report.update({"decision": decision, "failing_conditions": failures, "smallest_valid_scope_corrections": corrections})
    return report


def compact_print(report: dict[str, Any]) -> None:
    print("INSPECTIQ DAY 0 FEASIBILITY AUDIT")
    if "shapes" not in report:
        print("audit_status=data retrieval or cache failure")
        print(f"report={REPORT_PATH}")
        print(f"INSPECTIQ DAY 0 DECISION: {report['decision']}")
        return
    print(f"inspection={report['shapes']['inspection']} violation={report['shapes']['violation']} joined={report['shapes']['joined_inspection_level']}")
    print(f"dates={report['inspection_date_range']['min']} to {report['inspection_date_range']['max']} years={report['represented_year_count']} violation_retrieval_coverage={report['violation_retrieval_coverage']['percentage']}% label_coverage={report['label_coverage']['percentage']}%")
    print(f"labels: positive={report['label_counts']['positive']} negative={report['label_counts']['negative']} excluded_or_unknown={report['label_counts']['excluded_or_unknown']} positive_rate={report['label_counts']['positive_class_percentage']}")
    for warning in report.get("warnings", []):
        print(f"warning={warning['message']}")
    print(f"report={REPORT_PATH}")
    if report["failing_conditions"]:
        print("failing_conditions=" + ", ".join(report["failing_conditions"]))
    print(f"INSPECTIQ DAY 0 DECISION: {report['decision']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InspectIQ Day 0 feasibility audit")
    parser.add_argument("--state", default="CA")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached successful pages.")
    parser.add_argument("--offline", action="store_true", help="Use cached pages only; make no network requests.")
    parser.add_argument("--row-limit", type=int, default=MIN_USABLE_INSPECTIONS, help="Smaller limit is useful only for debugging.")
    parser.add_argument("--max-new-violation-batches", type=int, default=1, help="Maximum newly requested violation batches per run (default: 1).")
    parser.add_argument("--request-pause-seconds", type=float, default=VIOLATION_BATCH_PAUSE_SECONDS, help="Pause between newly requested violation batches.")
    args = parser.parse_args()
    if args.row_limit < 1:
        parser.error("--row-limit must be positive")
    if args.max_new_violation_batches < 0:
        parser.error("--max-new-violation-batches cannot be negative")
    if args.request_pause_seconds < 0:
        parser.error("--request-pause-seconds cannot be negative")
    try:
        if datetime.fromisoformat(args.start_date) > datetime.fromisoformat(args.end_date):
            parser.error("--start-date must not be after --end-date")
    except ValueError:
        parser.error("Dates must use YYYY-MM-DD.")
    return args


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        report = build_report(args)
    except (DOLApiError, OSError, ValueError) as error:
        attempt = {
            "audit": "InspectIQ Day 0 feasibility audit attempt",
            "generated_at": now_iso(), "configuration": vars(args),
            "error": str(error),
            "main_report_preserved": REPORT_PATH.exists(),
        }
        write_json(ATTEMPT_ERROR_PATH, attempt)
        print(f"attempt_error={ATTEMPT_ERROR_PATH}")
        print("INSPECTIQ DAY 0 DECISION: NO_GO")
        return None
    write_json(REPORT_PATH, report)
    compact_print(report)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

