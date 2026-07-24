from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BASE_URL = "https://apiprod.dol.gov/v4/get/OSHA"
OUTPUT_PATH = Path("reports/value_probe.json")
TIMEOUT_SECONDS = 90

INSPECTION_SAMPLE_LIMIT = 2000
STATE_INSPECTION_LIMIT = 500
JOIN_ID_LIMIT = 20


def load_api_key() -> str:
    load_dotenv(".env")

    key = os.getenv("DOL_API_KEY", "").strip()

    if not key:
        raise RuntimeError("DOL_API_KEY was not loaded from .env.")

    return key


def fetch_records(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    *,
    limit: int,
    offset: int = 0,
    fields: list[str] | None = None,
    sort_by: str | None = None,
    sort: str | None = None,
    filter_object: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "X-API-KEY": api_key,
        "limit": limit,
        "offset": offset,
    }

    if fields:
        params["fields"] = ",".join(fields)

    if sort_by:
        params["sort_by"] = sort_by

    if sort:
        params["sort"] = sort

    if filter_object:
        params["filter_object"] = json.dumps(
            filter_object,
            separators=(",", ":"),
        )

    url = f"{BASE_URL}/{endpoint}/json"

    response = session.get(
        url,
        params=params,
        timeout=TIMEOUT_SECONDS,
    )

    print(f"{endpoint}: HTTP {response.status_code}")

    # HTTP 204 means the request succeeded but no matching rows exist.
    if response.status_code == 204:
        return []

    # A rate-limited request is unknown, not an empty result.
    if response.status_code == 429:
        raise RuntimeError(
            "DOL API rate limit reached with HTTP 429."
        )

    if not response.ok:
        preview = response.text[:500].replace(
            api_key,
            "<REDACTED>",
        )
        raise RuntimeError(
            f"{endpoint} request failed with HTTP "
            f"{response.status_code}. Response: {preview}"
        )

    if not response.content.strip():
        return []

    try:
        payload = response.json()
    except ValueError as error:
        preview = response.text[:500].replace(
            api_key,
            "<REDACTED>",
        )
        raise RuntimeError(
            f"{endpoint} returned invalid JSON. Response: {preview}"
        ) from error
    records = payload.get("data", [])

    if not isinstance(records, list):
        raise ValueError(
            f"{endpoint} response did not contain a list under 'data'."
        )

    return [
        record
        for record in records
        if isinstance(record, dict)
    ]


def normalize_text(value: Any) -> str:
    if value is None:
        return "<NULL>"

    text = str(value).strip()
    return text if text else "<BLANK>"


def metadata_field_name(item: dict[str, Any]) -> str | None:
    possible_keys = (
        "field_name",
        "field",
        "name",
        "column_name",
        "variable_name",
    )

    for key in possible_keys:
        value = item.get(key)

        if value:
            return str(value)

    return None


def relevant_metadata(
    endpoint: str,
    wanted_fields: set[str],
) -> list[dict[str, Any]]:
    path = Path(f"reports/schema/{endpoint}_metadata.json")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, list):
        return []

    matches: list[dict[str, Any]] = []

    for item in payload:
        if not isinstance(item, dict):
            continue

        field_name = metadata_field_name(item)

        if field_name in wanted_fields:
            matches.append(item)

    return matches


def count_values(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counts = Counter(
        normalize_text(record.get(field))
        for record in records
    )

    return dict(counts.most_common())


def main() -> None:
    api_key = load_api_key()

    inspection_fields = [
        "activity_nr",
        "open_date",
        "case_mod_date",
        "close_case_date",
        "close_conf_date",
        "site_state",
        "mail_state",
        "naics_code",
        "sic_code",
        "insp_type",
        "insp_scope",
        "owner_type",
        "safety_hlth",
        "nr_in_estab",
        "estab_name",
    ]

    violation_fields = [
        "activity_nr",
        "citation_id",
        "viol_type",
        "delete_flag",
        "rec",
        "issuance_date",
        "initial_penalty",
        "current_penalty",
        "contest_date",
        "final_order_date",
    ]

    report: dict[str, Any] = {
        "inspection_sample_limit": INSPECTION_SAMPLE_LIMIT,
        "state_inspection_limit": STATE_INSPECTION_LIMIT,
        "join_id_limit": JOIN_ID_LIMIT,
    }

    print("=" * 76)
    print("OSHA VALUE AND JOIN PROBE")
    print("=" * 76)

    with requests.Session() as session:
        recent_inspections = fetch_records(
            session,
            "inspection",
            api_key,
            limit=INSPECTION_SAMPLE_LIMIT,
            fields=inspection_fields,
            sort_by="open_date",
            sort="desc",
        )

        if not recent_inspections:
            raise RuntimeError("No inspection records were returned.")

        state_counts = count_values(
            recent_inspections,
            "site_state",
        )

        valid_state_counts = {
            state: count
            for state, count in state_counts.items()
            if state not in {"<NULL>", "<BLANK>"}
        }

        if not valid_state_counts:
            raise RuntimeError(
                "No usable site_state values were found."
            )

        candidate_state = max(
            valid_state_counts,
            key=valid_state_counts.get,
        )

        # Use a mature historical window so inspection outcomes
        # have had sufficient time to appear in the violation dataset.
        historical_start = "2023-12-31T23:59:59"
        historical_end = "2025-01-01T00:00:00"

        state_filter = {
            "and": [
                {
                    "field": "site_state",
                    "operator": "eq",
                    "value": candidate_state,
                },
                {
                    "field": "open_date",
                    "operator": "gt",
                    "value": historical_start,
                },
                {
                    "field": "open_date",
                    "operator": "lt",
                    "value": historical_end,
                },
            ]
        }

        state_inspections = fetch_records(
            session,
            "inspection",
            api_key,
            limit=STATE_INSPECTION_LIMIT,
            fields=inspection_fields,
            sort_by="open_date",
            sort="desc",
            filter_object=state_filter,
        )

        activity_ids = [
            normalize_text(record.get("activity_nr"))
            for record in state_inspections
            if normalize_text(record.get("activity_nr"))
            not in {"<NULL>", "<BLANK>"}
        ]

        unique_activity_ids = list(dict.fromkeys(activity_ids))
        join_ids = unique_activity_ids[:JOIN_ID_LIMIT]

        if not join_ids:
            raise RuntimeError(
                "No usable activity_nr values were found."
            )

        matching_violations: list[dict[str, Any]] = []

        print()
        print("FETCHING MATCHING VIOLATIONS")
        print("-" * 76)

        batch_size = 5
        batches = [
            join_ids[start : start + batch_size]
            for start in range(0, len(join_ids), batch_size)
        ]

        for batch_number, batch_ids in enumerate(batches, start=1):
            violation_filter = {
                "field": "activity_nr",
                "operator": "in",
                "value": batch_ids,
            }

            rows: list[dict[str, Any]] = []

            for attempt in range(1, 6):
                try:
                    rows = fetch_records(
                        session,
                        "violation",
                        api_key,
                        limit=5000,
                        fields=violation_fields,
                        filter_object=violation_filter,
                    )
                    break
                except RuntimeError as error:
                    is_rate_limit = "rate limit" in str(error).lower()

                    if not is_rate_limit or attempt == 5:
                        raise

                    wait_seconds = 2 ** attempt
                    print(
                        f"Rate limited. Waiting {wait_seconds} seconds "
                        f"before retry {attempt + 1}/5."
                    )
                    time.sleep(wait_seconds)

            matching_violations.extend(rows)

            print(
                f"Batch {batch_number}/{len(batches)}: "
                f"{len(batch_ids)} inspection IDs, "
                f"{len(rows)} violation rows"
            )

            time.sleep(1)

    violation_activity_ids = {
        normalize_text(record.get("activity_nr"))
        for record in matching_violations
    }

    joined_activity_ids = {
        activity_id
        for activity_id in join_ids
        if activity_id in violation_activity_ids
    }

    inspection_dates = [
        normalize_text(record.get("open_date"))
        for record in recent_inspections
        if normalize_text(record.get("open_date"))
        not in {"<NULL>", "<BLANK>"}
    ]

    report.update(
        {
            "recent_inspection_rows": len(recent_inspections),
            "recent_open_date_first": (
                inspection_dates[0]
                if inspection_dates
                else None
            ),
            "recent_open_date_last": (
                inspection_dates[-1]
                if inspection_dates
                else None
            ),
            "site_state_counts": state_counts,
            "temporary_candidate_state": candidate_state,
            "candidate_state_rows_returned": len(state_inspections),
            "candidate_state_unique_activity_ids": len(
                unique_activity_ids
            ),
            "activity_ids_tested_for_join": len(join_ids),
            "matching_violation_rows": len(matching_violations),
            "activity_ids_with_at_least_one_violation": len(
                joined_activity_ids
            ),
            "activity_id_join_rate_percentage": round(
                100 * len(joined_activity_ids) / len(join_ids),
                2,
            ),
            "inspection_activity_nr_duplicates": (
                len(activity_ids) - len(unique_activity_ids)
            ),
            "viol_type_counts": count_values(
                matching_violations,
                "viol_type",
            ),
            "delete_flag_counts": count_values(
                matching_violations,
                "delete_flag",
            ),
            "rec_counts": count_values(
                matching_violations,
                "rec",
            ),
            "naics_missing_counts": count_values(
                state_inspections,
                "naics_code",
            ),
            "sic_missing_counts": count_values(
                state_inspections,
                "sic_code",
            ),
            "inspection_metadata": relevant_metadata(
                "inspection",
                {
                    "activity_nr",
                    "open_date",
                    "case_mod_date",
                    "close_case_date",
                    "close_conf_date",
                    "site_state",
                    "naics_code",
                    "sic_code",
                    "insp_type",
                    "nr_in_estab",
                },
            ),
            "violation_metadata": relevant_metadata(
                "violation",
                {
                    "activity_nr",
                    "viol_type",
                    "delete_flag",
                    "rec",
                    "issuance_date",
                    "initial_penalty",
                    "current_penalty",
                    "contest_date",
                    "final_order_date",
                },
            ),
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("RECENT INSPECTION SAMPLE")
    print("-" * 76)
    print(f"Rows returned:       {len(recent_inspections)}")
    print(
        "Newest open_date:    "
        f"{report['recent_open_date_first']}"
    )
    print(
        "Oldest open_date:    "
        f"{report['recent_open_date_last']}"
    )

    print()
    print("TOP SITE STATES")
    print("-" * 76)

    for state, count in list(state_counts.items())[:15]:
        print(f"{state}: {count}")

    print()
    print("TEMPORARY STATE CANDIDATE")
    print("-" * 76)
    print(f"State:                     {candidate_state}")
    print(f"Inspection rows:           {len(state_inspections)}")
    print(f"Unique activity_nr values: {len(unique_activity_ids)}")
    print(
        "Duplicate activity_nr rows: "
        f"{report['inspection_activity_nr_duplicates']}"
    )

    print()
    print("JOIN TEST")
    print("-" * 76)
    print(f"Inspection IDs tested: {len(join_ids)}")
    print(f"Violation rows found:  {len(matching_violations)}")
    print(
        "IDs with violations:  "
        f"{len(joined_activity_ids)}"
    )
    print(
        "Join rate:            "
        f"{report['activity_id_join_rate_percentage']}%"
    )

    print()
    print("VIOLATION TYPE COUNTS")
    print("-" * 76)

    for value, count in report["viol_type_counts"].items():
        print(f"{value}: {count}")

    print()
    print("DELETE FLAG COUNTS")
    print("-" * 76)

    for value, count in report["delete_flag_counts"].items():
        print(f"{value}: {count}")

    print()
    print("REC FIELD COUNTS")
    print("-" * 76)

    for value, count in report["rec_counts"].items():
        print(f"{value}: {count}")

    print()
    print(f"Saved report: {OUTPUT_PATH.resolve()}")
    print("=" * 76)
    print("VALUE AND JOIN PROBE COMPLETED")
    print("=" * 76)


if __name__ == "__main__":
    main()
