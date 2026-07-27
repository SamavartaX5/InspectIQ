"""Command-line entrypoint for Day 5B offline candidate ranking."""
from pathlib import Path

from src.batch_prediction import PredictionError, atomic_json, run


def main() -> None:
    try:
        report = run()
        atomic_json(Path("reports/batch_prediction_report.json"), report)
    except Exception as exc:
        atomic_json(Path("reports/batch_prediction_attempt_error.json"), {"status": "FAIL", "error": str(exc)})
        print(f"INSPECTIQ BATCH PREDICTION ERROR: {exc}")
        print("INSPECTIQ BATCH PREDICTION: FAIL")
        return
    print(
        f"candidates={report['input_row_count']} model={report['model_artifact_path']} "
        f"selected_calibration={report['selected_day4_method']} "
        f"score_min={report['score_summary']['minimum']:.6f} score_max={report['score_summary']['maximum']:.6f} "
        f"top_5={report['top_counts']['top_5_percent']} top_10={report['top_counts']['top_10_percent']} "
        f"top_20={report['top_counts']['top_20_percent']} ranked={report['ranked_output_path']} "
        f"top_10_output={report['top_10_output_path']} artifacts_reused={str(report['artifacts_reused']).lower()} "
        "labels_accessed=false performance_metrics_calculated=false automatic_enforcement=false"
    )
    print("INSPECTIQ BATCH PREDICTION: PASS")


if __name__ == "__main__":
    main()
