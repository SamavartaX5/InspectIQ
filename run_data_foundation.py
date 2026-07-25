"""Run the cache-only InspectIQ Day 1 data-foundation pipeline."""

from __future__ import annotations

from pathlib import Path

from src.data_foundation import FoundationError, run_foundation, write_json


def main() -> None:
    try:
        report = run_foundation(
            report_path=Path("reports/feasibility_report.json"), schema_path=Path("config/schema.yaml"),
            snapshot_root=Path("data/raw/snapshots"), processed_root=Path("data/processed"),
        )
        write_json(Path("reports/data_foundation_report.json"), report)
    except FoundationError as error:
        print(f"INSPECTIQ DATA FOUNDATION ERROR: {error}")
        print("INSPECTIQ DATA FOUNDATION: FAIL")
        return
    print(f"snapshot_id={report['snapshot_id']} labelled={report['label_counts']['labelled']} excluded={report['excluded_table_shape'][0]} rejected={report['rejected_table_shape'][0]}")
    print("INSPECTIQ DATA FOUNDATION: PASS")


if __name__ == "__main__":
    main()
