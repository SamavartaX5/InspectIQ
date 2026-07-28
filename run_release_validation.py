"""Run read-only Day 7A release validation against existing artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.release_validation import ReleaseValidationError, validate_release, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description="InspectIQ read-only release validation")
    parser.add_argument("--mode", choices=("ci", "local"), required=True, help="ci omits ignored local artifacts; local validates frozen artifacts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report_path = root / "reports" / "release_validation_report.json"
    error_path = root / "reports" / "release_validation_attempt_error.json"
    try:
        report = validate_release(root, args.mode)
        write_json_atomic(report_path, report)
    except Exception as exc:
        error = {
            "status": "FAIL",
            "mode": args.mode,
            "error": str(exc),
            "limitations": ["The prior valid release validation report was preserved."],
        }
        write_json_atomic(error_path, error)
        print(f"INSPECTIQ RELEASE VALIDATION {args.mode.upper()}: FAIL")
        print(f"reason={exc}")
        return 1
    print(f"mode={args.mode}")
    print("required_files_valid=true configs_valid=true reports_valid=true")
    print(f"artifacts_valid={'true' if args.mode == 'local' else 'not_required'}")
    print("docker_contract_valid=true ci_workflow_valid=true streamlit_import_valid=true")
    print("labels_accessed=false performance_metrics_calculated=false outcome_fairness_metrics_calculated=false automatic_enforcement=false")
    print(f"INSPECTIQ RELEASE VALIDATION {args.mode.upper()}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
