from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


CATALOG_URL = "https://apiprod.dol.gov/v4/datasets"
OUTPUT_PATH = Path("reports/dol_osha_catalog.json")
TIMEOUT_SECONDS = 30


def fetch_catalog() -> list[dict[str, Any]]:
    """Download every page of the public DOL dataset catalog."""

    all_datasets: list[dict[str, Any]] = []
    page = 1

    with requests.Session() as session:
        while True:
            response = session.get(
                CATALOG_URL,
                params={"page": page},
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            payload = response.json()
            datasets = payload.get("datasets", [])
            metadata = payload.get("meta", {})

            if not isinstance(datasets, list):
                raise ValueError("DOL catalog response did not contain a dataset list.")

            all_datasets.extend(datasets)

            next_page = metadata.get("next_page")
            if next_page is None:
                break

            page = int(next_page)

    return all_datasets


def is_osha_dataset(dataset: dict[str, Any]) -> bool:
    agency = dataset.get("agency") or {}
    agency_abbreviation = str(agency.get("abbr", "")).upper()
    agency_name = str(agency.get("name", "")).lower()

    return (
        agency_abbreviation == "OSHA"
        or "occupational safety and health administration" in agency_name
    )


def is_relevant_dataset(dataset: dict[str, Any]) -> bool:
    searchable_text = " ".join(
        [
            str(dataset.get("name", "")),
            str(dataset.get("description", "")),
            str(dataset.get("api_url", "")),
            " ".join(str(tag) for tag in dataset.get("tag_list", [])),
        ]
    ).lower()

    terms = (
        "inspection",
        "violation",
        "citation",
        "establishment",
    )

    return any(term in searchable_text for term in terms)


def main() -> None:
    all_datasets = fetch_catalog()
    osha_datasets = [item for item in all_datasets if is_osha_dataset(item)]
    relevant_datasets = [
        item for item in osha_datasets if is_relevant_dataset(item)
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(osha_datasets, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("DOL DATASET CATALOG DISCOVERY")
    print("=" * 72)
    print(f"All catalog datasets found: {len(all_datasets)}")
    print(f"OSHA datasets found:       {len(osha_datasets)}")
    print(f"Relevant OSHA datasets:    {len(relevant_datasets)}")
    print(f"Saved catalog snapshot:    {OUTPUT_PATH.resolve()}")
    print()

    if not relevant_datasets:
        print("No relevant OSHA inspection/violation datasets were discovered.")
        return

    print("POTENTIALLY RELEVANT OSHA DATASETS")
    print("-" * 72)

    for dataset in sorted(
        relevant_datasets,
        key=lambda item: str(item.get("name", "")).lower(),
    ):
        print(f"Name:        {dataset.get('name')}")
        print(f"API endpoint:{dataset.get('api_url')}")
        print(f"Table:       {dataset.get('tablename')}")
        print(f"Frequency:   {dataset.get('frequency')}")
        print(f"Published:   {dataset.get('published_at')}")
        print(f"Updated:     {dataset.get('updated_at')}")
        print(f"Description: {dataset.get('description')}")
        print("-" * 72)


if __name__ == "__main__":
    main()
