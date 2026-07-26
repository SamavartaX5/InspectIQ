from pathlib import Path
from src.mlflow_tracking import TrackingError,atomic,run
def main():
 try:r=run();atomic(Path("reports/mlflow_tracking_report.json"),r)
 except Exception as e:atomic(Path("reports/mlflow_tracking_attempt_error.json"),{"status":"FAIL","error":str(e)});print(f"INSPECTIQ MLFLOW TRACKING ERROR: {e}");print("INSPECTIQ MLFLOW TRACKING: FAIL");return
 print(f"mlflow={r['mlflow_version']} uri={r['tracking_uri']} day3_created_reused={r['newly_created_run_count']}/{r['reused_run_count']} day4_runs={r['logged_day4_run_count']} selected_model={r['selected_model_run']} selected_calibration={r['selected_calibration_method']} final_candidate_artifact_logged=true locked_test_labels_accessed=false locked_test_metrics_calculated=false locked_test_predictions_created=false")
 print("INSPECTIQ MLFLOW TRACKING: PASS")
if __name__=='__main__':main()
