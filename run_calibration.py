from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from src.calibration import CalibrationError,calibrate,metrics,select_method
from src.training import TARGET,CATEGORICAL,NUMERIC,load_inputs,pipeline,read_json
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); t=path.with_suffix(path.suffix+".tmp");t.write_text(json.dumps(obj,indent=2,default=str),encoding="utf8");t.replace(path)
def run():
 cfg=read_json(Path("config/calibration_config.yaml")); fr=read_json(Path("reports/feature_engineering_report.json")); d3=read_json(Path("reports/model_comparison_report.json")); frames,manifest=load_inputs(Path(fr["output_paths"]["manifest"])); spec=next(x["hyperparameters"] for x in d3["experiments"] if x["experiment_id"]==cfg["selected_experiment_id"])
 train=frames["train"]; cal=train[pd.to_datetime(train.open_date).dt.year==2021]; base=train[pd.to_datetime(train.open_date).dt.year==2020]; val=frames["validation"]
 if min(len(base),len(cal))<cfg["minimum_calibration_rows"] or min(cal[TARGET].sum(),len(cal)-cal[TARGET].sum())<cfg["minimum_calibration_positives"]: raise CalibrationError("Chronological calibration period does not meet minimum class requirements.")
 x=lambda d:d[CATEGORICAL+NUMERIC]; y=lambda d:d[TARGET].astype(int).to_numpy(); model=pipeline(spec);model.fit(x(base),y(base)); raw=model.predict_proba(x(val))[:,1]
 methods={"uncalibrated":(model,raw)}
 for method in ("sigmoid","isotonic"):
  c=calibrate(model,x(cal),y(cal),method);methods[method]=(c,c.predict_proba(x(val))[:,1])
 runid=hashlib.sha256(json.dumps({"hashes":manifest["file_hashes"],"cfg":cfg},sort_keys=True).encode()).hexdigest()[:16];root=Path("artifacts/models/day4")/runid;root.mkdir(parents=True,exist_ok=True);results={}
 for name,(obj,p) in methods.items():
  ranked,m=metrics(p,y(val),val.activity_nr.astype(str),cfg["reliability_bin_count"]);joblib.dump(obj,root/"study"/f"{name}.joblib") if (root/"study").mkdir(parents=True,exist_ok=True) is None else None;results[name]=m
 chosen,improved,rationale=select_method(results,cfg)
 final=pipeline(spec);final.fit(x(train),y(train)); packaged=final if chosen=="uncalibrated" else calibrate(final,x(val),y(val),chosen);(root/"final").mkdir(exist_ok=True);final_name="final_candidate.joblib";joblib.dump(packaged,root/"final"/final_name)
 report={"status":"PASS","source_snapshot_id":manifest["source_snapshot_id"],"feature_version":manifest["feature_version"],"day3_training_run_id":d3["training_run_id"],"calibration_run_id":runid,"selected_day3_experiment":cfg["selected_experiment_id"],"selected_hyperparameters":spec,"periods":{"base_fit":{"rows":len(base),"dates":[str(base.open_date.min()),str(base.open_date.max())],"positives":int(y(base).sum())},"calibration":{"rows":len(cal),"dates":[str(cal.open_date.min()),str(cal.open_date.max())],"positives":int(y(cal).sum())},"validation":{"rows":len(val),"dates":[str(val.open_date.min()),str(val.open_date.max())],"positives":int(y(val).sum())},"strictly_chronological":True},"locked_test_row_count":len(frames["test"]),"locked_test_labels_accessed":False,"locked_test_metrics_calculated":False,"locked_test_predictions_created":False,"study_results":results,"selected_calibration_method":chosen,"calibration_improved_probability_quality":improved,"selection_rationale":"Lowest Brier, then log loss, ECE, Recall@10 tolerance; sigmoid preferred on ties.","original_day3_raw_brier":next(e["metrics"]["brier_score"] for e in d3["experiments"] if e["experiment_id"]==cfg["selected_experiment_id"]),"study_artifact_paths":{n:str(root/"study"/f"{n}.joblib") for n in methods},"final_calibrated_artifact_path":str(root/"final"/"calibrated_candidate.joblib"),"final_package_fit_period":"2020-2021","final_package_calibration_period":"2022","final_package_performance_reported":False,"limitations":["Calibration results are retrospective.","The calibration study uses a smaller chronological base-model training period than Day 3.","Isotonic calibration can overfit when calibration data is limited.","Sigmoid calibration imposes a parametric shape.","Raw and calibrated probabilities do not establish causality.","The locked test remains untouched."]};write(root/"final"/"calibration_manifest.json",report);return report
def main():
 try:
  r=run()
  r["selection_rationale"] = "Calibration was evaluated but rejected because neither calibrated method meaningfully improved validation Brier score while preserving Recall@10 tolerance." if r["selected_calibration_method"] == "uncalibrated" else r["selection_rationale"]
  r["final_candidate_artifact_path"] = str(Path("artifacts/models/day4") / r["calibration_run_id"] / "final" / ("final_candidate.joblib" if r["selected_calibration_method"] == "uncalibrated" else "calibrated_candidate.joblib"))
  r.pop("final_calibrated_artifact_path", None)
  r["final_calibration_applied"] = r["selected_calibration_method"] != "uncalibrated"
  r["final_package_type"] = r["selected_calibration_method"]
  if not r["final_calibration_applied"]: r["final_package_calibration_period"] = "not applicable"
  r["final_candidate_artifact_path"] = str(Path("artifacts/models/day4") / r["calibration_run_id"] / "final" / "final_candidate.joblib")
  write(Path(r["final_candidate_artifact_path"]).parent / "calibration_manifest.json", r)
  write(Path("reports/calibration_report.json"),r)
 except Exception as e:write(Path("reports/calibration_attempt_error.json"),{"status":"FAIL","error":str(e)});print(f"INSPECTIQ CALIBRATION ERROR: {e}");print("INSPECTIQ CALIBRATION: FAIL");return
 print(f"base={r['periods']['base_fit']['rows']} calibration={r['periods']['calibration']['rows']} validation={r['periods']['validation']['rows']}");print(f"selected={r['selected_calibration_method']} improved={r['calibration_improved_probability_quality']} final={r['final_candidate_artifact_path']} locked_test_labels_accessed=false locked_test_metrics_calculated=false locked_test_predictions_created=false");print("INSPECTIQ CALIBRATION: PASS")
if __name__=='__main__':main()
