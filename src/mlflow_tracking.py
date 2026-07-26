"""Local-only tracking of existing InspectIQ evidence; it never trains or scores."""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
from typing import Any
import mlflow
from mlflow.tracking import MlflowClient
class TrackingError(RuntimeError): pass
def read(path:Path): return json.loads(path.read_text(encoding="utf8"))
def digest(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path:Path,value):
 t=path.with_suffix(path.suffix+".tmp");t.write_text(json.dumps(value,indent=2),encoding="utf8");t.replace(path)
def finite(value):
 if not isinstance(value,(int,float)) or not math.isfinite(value): raise TrackingError(f"Non-finite metric: {value}")
def flatten_metrics(metrics):
 out={}
 for p in ("5","10","20"):
  row=metrics["ranking_at"][p];out.update({f"recall_at_{p}":row["recall"],f"precision_at_{p}":row["precision"],f"lift_at_{p}":row["lift"]})
 out.update({"positives_captured_top_10":metrics["ranking_at"]["10"]["selected_positives"],"pr_auc":metrics["pr_auc"],"roc_auc":metrics["roc_auc"],"brier_score":metrics["brier_score"]})
 for n,v in metrics.get("threshold_0_5",{}).items():
  if n!="confusion_matrix":out[f"threshold_0_5_{n}"]=v
 return out
def validate(day3,day4):
 if day3.get("status")!="PASS" or day4.get("status")!="PASS":raise TrackingError("Source reports must PASS.")
 if day3["source_snapshot_id"]!=day4["source_snapshot_id"] or day3["feature_version"]!=day4["feature_version"]:raise TrackingError("Snapshot or feature versions are incompatible.")
 if any(day3.get(k) for k in ("locked_test_labels_accessed","locked_test_metrics_calculated")) or any(day4.get(k) for k in ("locked_test_labels_accessed","locked_test_metrics_calculated","locked_test_predictions_created")):raise TrackingError("Locked-test safeguard violation.")
 if not Path(day4["final_candidate_artifact_path"]).exists():raise TrackingError("Final candidate artifact is missing.")
def run(day3_path=Path("reports/model_comparison_report.json"),day4_path=Path("reports/calibration_report.json"),config_path=Path("config/mlflow_config.yaml")):
 started=time.perf_counter();d3=read(day3_path);d4=read(day4_path);cfg=read(config_path);validate(d3,d4);batch=hashlib.sha256(json.dumps({"d3":digest(day3_path),"d4":digest(day4_path),"cfg":digest(config_path),"snapshot":d3["source_snapshot_id"],"training":d3["training_run_id"],"calibration":d4["calibration_run_id"]},sort_keys=True).encode()).hexdigest()[:20]
 mlflow.set_tracking_uri(cfg["tracking_uri"]);client=MlflowClient();root=(Path(cfg["artifact_root"]).resolve()).as_uri();ids={}
 for kind,name in (("day3",cfg["day3_experiment"]),("day4",cfg["day4_experiment"])):
  exp=client.get_experiment_by_name(name);ids[kind]=exp.experiment_id if exp else client.create_experiment(name,artifact_location=root+f"/{kind}")
 created=reused=0;records=[]
 def log(kind,key,params,metrics,tags,files):
  nonlocal created,reused
  found=client.search_runs([ids[kind]],filter_string=f"tags.logical_run_key = '{key}' and tags.tracking_batch_id = '{batch}'",max_results=10)
  if found:
   records.append({"logical_run_key":key,"mlflow_run_id":found[0].info.run_id,"reused":True});reused+=1;return
  with mlflow.start_run(experiment_id=ids[kind],run_name=key) as active:
   mlflow.set_tags({**{k:str(v).lower() if isinstance(v,bool) else str(v) for k,v in tags.items()},"logical_run_key":key,"tracking_batch_id":batch})
   mlflow.log_params({k:str(v) for k,v in params.items() if v is not None})
   for k,v in metrics.items():finite(v);mlflow.log_metric(k,float(v))
   for file in files:
    if not Path(file).exists():raise TrackingError(f"Required artifact missing: {file}")
    mlflow.log_artifact(str(file))
   records.append({"logical_run_key":key,"mlflow_run_id":active.info.run_id,"reused":False});created+=1
 for exp in d3["experiments"]:
  m=flatten_metrics(exp["metrics"]);m.update({"training_runtime_seconds":exp["training_runtime_seconds"],"validation_scoring_runtime_seconds":exp["validation_scoring_runtime_seconds"],"artifact_size_bytes":exp["artifact_size_bytes"]});params={"experiment_id":exp["experiment_id"],"model_name":exp["model_name"],**exp["hyperparameters"],"source_snapshot_id":d3["source_snapshot_id"],"feature_version":d3["feature_version"],"training_run_id":d3["training_run_id"],"train_rows":d3["train_shape"][0],"validation_rows":d3["validation_shape"][0]};files=[exp["artifact_path"],str(Path(exp["artifact_path"]).parent/"validation_scoring.csv"),str(day3_path)];log("day3",f"day3:{exp['experiment_id']}:{batch}",params,m,{"selected_candidate":exp["experiment_id"]==d3["selected_candidate"],"ml_value_added":d3["ml_value_added"],"retrospective_evaluation":True,"chronological_validation":True,"locked_test_labels_accessed":False,"locked_test_metrics_calculated":False,"human_review_required":True,"automated_enforcement":False},files)
 for method,m in d4["study_results"].items():
  mets=flatten_metrics(m);mets.update({k:m[k] for k in ("log_loss","expected_calibration_error","maximum_calibration_error","mean_predicted_probability","observed_positive_rate","mean_probability_gap")});files=[d4["study_artifact_paths"][method],str(day4_path)];
  if method==d4["selected_calibration_method"]:files.append(d4["final_candidate_artifact_path"])
  log("day4",f"day4:{method}:{batch}",{"calibration_method":method,"selected_day3_experiment":d4["selected_day3_experiment"],"source_snapshot_id":d4["source_snapshot_id"],"feature_version":d4["feature_version"],"calibration_run_id":d4["calibration_run_id"]},mets,{"selected_calibration_method":method==d4["selected_calibration_method"],"calibration_improved_probability_quality":d4["calibration_improved_probability_quality"],"final_calibration_applied":d4["final_calibration_applied"],"retrospective_evaluation":True,"locked_test_labels_accessed":False,"locked_test_metrics_calculated":False,"locked_test_predictions_created":False},files)
 return {"status":"PASS","mlflow_version":mlflow.__version__,"tracking_uri":cfg["tracking_uri"],"artifact_root":cfg["artifact_root"],"tracking_batch_id":batch,"experiments":ids,"expected_day3_run_count":len(d3["experiments"]),"logged_day3_run_count":len(d3["experiments"]),"expected_day4_run_count":len(d4["study_results"]),"logged_day4_run_count":len(d4["study_results"]),"newly_created_run_count":created,"reused_run_count":reused,"runs":records,"selected_model_run":d3["selected_candidate"],"selected_calibration_method":d4["selected_calibration_method"],"final_candidate_artifact_logged":True,"report_artifacts_logged":True,"locked_test_labels_accessed":False,"locked_test_metrics_calculated":False,"locked_test_predictions_created":False,"runtime_seconds":round(time.perf_counter()-started,4),"limitations":["Local retrospective evidence tracking only; no models were trained or scored."]}
