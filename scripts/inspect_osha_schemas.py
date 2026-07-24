from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BASE_URL = "https://apiprod.dol.gov/v4/get/OSHA"
ENDPOINTS = ("inspection", "violation")
OUTPUT_DIR = Path("reports/schema")
SAMPLE_LIMIT = 10
TIMEOUT_SECONDS = 60


def load_api_key() -> str:
    load_dotenv(".env")

    api_key = os.getenv("DOL_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "DOL_API_KEY could not be loaded from the .env file."
        )

    return api_key


def fetch_json(
    session: requests.Session,
    url: str,
    api_key: str,
    **parameters: Any,
) -> Any:
    params = {
        "X-API-KEY": api_key,
        **parameters,
    }

    response = session.get(
        url,
        params=params,
        timeout=TIMEOUT_SECONDS,
    )

    print(f"Request: {url}")
    print(f"HTTP status: {response.status_code}")

    if response.status_code in {401, 403}:
        raise RuntimeError(
            "Authentication failed. Verify that the DOL API key is active."
        )

    response.raise_for_status()

    try:
        return response.json()
    except ValueError as error:
        preview = response.text[:500]
        raise RuntimeError(
            f"The API returned non-JSON content: {preview}"
        ) from error


def find_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return payload
        return []

    if not isinstance(payload, dict):
        return []

    preferred_keys = (
        "data",
        "results",
        "records",
        "items",
    )

    for key in preferred_keys:
        value = payload.get(key)

        if isinstance(value, list) and all(
            isinstance(item, dict) for item in value
        ):
            return value

    for value in payload.values():
        records = find_records(value)

        if records:
            return records

    return []


def describe_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        return f"dictionary with keys: {sorted(payload.keys())}"

    if isinstance(payload, list):
        return f"list containing {len(payload)} items"

    return type(payload).__name__


def inspect_endpoint(
    session: requests.Session,
    endpoint: str,
    api_key: str,
) -> None:
    metadata_url = f"{BASE_URL}/{endpoint}/json/metadata"
    records_url = f"{BASE_URL}/{endpoint}/json"

    print()
    print("=" * 76)
    print(f"INSPECTING ENDPOINT: {endpoint}")
    print("=" * 76)

    metadata = fetch_json(
        session,
        metadata_url,
        api_key,
    )

    sample_payload = fetch_json(
        session,
        records_url,
        api_key,
        limit=SAMPLE_LIMIT,
        offset=0,
    )

    metadata_path = OUTPUT_DIR / f"{endpoint}_metadata.json"
    sample_path = OUTPUT_DIR / f"{endpoint}_sample.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    sample_path.write_text(
        json.dumps(sample_payload, indent=2),
        encoding="utf-8",
    )

    records = find_records(sample_payload)

    columns = sorted(
        {
            str(column)
            for record in records
            for column in record.keys()
        }
    )

    print()
    print(f"Metadata structure: {describe_payload(metadata)}")
    print(f"Sample structure:   {describe_payload(sample_payload)}")
    print(f"Records extracted:  {len(records)}")
    print(f"Columns discovered: {len(columns)}")
    print()

    print("COLUMNS")
    print("-" * 76)

    if columns:
        for column in columns:
            print(column)
    else:
        print("No record columns were extracted.")

    print()
    print(f"Saved: {metadata_path.resolve()}")
    print(f"Saved: {sample_path.resolve()}")


def main() -> None:
    api_key = load_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("OSHA SCHEMA INSPECTION")
    print("=" * 76)
    print("Endpoints: inspection and violation")
    print(f"Requested records per endpoint: {SAMPLE_LIMIT}")
    print("The API key will not be displayed.")

    with requests.Session() as session:
        for endpoint in ENDPOINTS:
            inspect_endpoint(
                session=session,
                endpoint=endpoint,
                api_key=api_key,
            )

    print()
    print("=" * 76)
    print("SCHEMA INSPECTION COMPLETED")
    print("=" * 76)


if __name__ == "__main__":
    main()
