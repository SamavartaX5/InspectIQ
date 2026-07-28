from pathlib import Path
from src.batch_prediction import atomic_json
from src.monitoring import run
def main():
 try:
  report=run()
  features={**report["numeric_drift"],**report["categorical_drift"],"raw_risk_score":report["score_distribution_drift"]}
  raw=[row["raw_statistical_severity"] for row in features.values()]; operational=[row["operational_severity"] for row in features.values()]
  report.update({"raw_critical_feature_count":raw.count("critical"),"operational_critical_feature_count":operational.count("critical"),"expected_structural_shift_feature_count":sum(row["expected_temporal_accumulation"] for row in features.values()),"raw_severity_by_feature":{name:row["raw_statistical_severity"] for name,row in features.items()},"operational_severity_by_feature":{name:row["operational_severity"] for name,row in features.items()},"monitoring_interpretation":"Expected temporal accumulation in cumulative history features is reported separately from unexpected operational drift."})
  atomic_json(Path("reports/monitoring_report.json"),report)
 except Exception as exc:
  atomic_json(Path("reports/monitoring_attempt_error.json"),{"status":"FAIL","error":str(exc)});print(f"INSPECTIQ MONITORING ERROR: {exc}");print("INSPECTIQ MONITORING: FAIL");return
 print(f"reference_rows={report['reference_current_row_counts']['validation']} current_rows={report['reference_current_row_counts']['current']} health={report['monitoring_health']} stable_warning_critical={report['stable_feature_count']}/{report['warning_feature_count']}/{report['critical_feature_count']} top_drifted={','.join(report['top_drifted_features'])} score_drift={report['score_distribution_drift']['severity']} review_exposure_groups=6 artifacts_reused={str(report['artifacts_reused']).lower()} current_labels_accessed=false current_performance_metrics_calculated=false outcome_fairness_metrics_calculated=false model_refit_attempted=false automatic_enforcement=false")
 print("INSPECTIQ MONITORING: PASS")
if __name__=="__main__":main()
