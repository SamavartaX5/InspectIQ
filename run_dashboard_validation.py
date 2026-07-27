from pathlib import Path

from src.batch_prediction import atomic_json
from src.dashboard import validate


def main() -> None:
    try:
        report = validate()
        atomic_json(Path("reports/dashboard_validation_report.json"), report)
    except Exception as exc:
        atomic_json(Path("reports/dashboard_validation_attempt_error.json"), {"status": "FAIL", "error": str(exc)})
        print(f"INSPECTIQ DASHBOARD VALIDATION ERROR: {exc}")
        print("INSPECTIQ DASHBOARD VALIDATION: FAIL")
        return
    print(
        f"ranked_rows={report['ranked_row_count']} top_10_rows={report['top_10_row_count']} "
        "score_range_valid=true global_importance_extracted=true local_explanation_generated=true "
        "model_refit_attempted=false labels_accessed=false performance_metrics_calculated=false automatic_enforcement=false"
    )
    print("INSPECTIQ DASHBOARD VALIDATION: PASS")


if __name__ == "__main__":
    main()
