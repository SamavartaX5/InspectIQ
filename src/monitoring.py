"""Offline, outcome-free Day 6 data/score drift and review-exposure monitoring."""
from __future__ import annotations
import hashlib, json, math, shutil, time, uuid
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, wasserstein_distance
from src.batch_prediction import MODEL_COLUMNS, PROHIBITED_COLUMNS, TARGET, atomic_json
from src.governance import future_outcome_template, review_template

CATEGORICAL=["naics_group","insp_type","insp_scope","owner_type","safety_hlth","industry_history_status"]
NUMERIC=["nr_in_estab","open_month","industry_prior_inspection_count","industry_prior_positive_count","industry_prior_positive_rate_smoothed"]
GROUPS=CATEGORICAL
class MonitoringError(RuntimeError): pass
def digest(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path:Path):
 try:return json.loads(path.read_text(encoding="utf8"))
 except Exception as exc:raise MonitoringError(f"Invalid JSON artifact: {path}") from exc
def config(path:Path):
 try:return yaml.safe_load(path.read_text(encoding="utf8"))
 except Exception as exc:raise MonitoringError(f"Invalid monitoring config: {path}") from exc
def sev(value,warn,critical): return "critical" if value>=critical else "warning" if value>=warn else "stable"
def operational_severity(feature,result,raw,cfg):
 """Keep raw drift visible; only expected cumulative growth can cap operations."""
 kind=cfg["feature_semantics"].get(feature,"standard_input"); missing=sev(result.get("missingness_change_pp",0),cfg["missingness_increase_pp"]["warning"],cfg["missingness_increase_pp"]["critical"]); reasons=[]
 valid=result.get("data_contract_valid",True) and result.get("values_finite",True) and result.get("reference_minimum",0)>=0 and result.get("current_minimum",0)>=0
 mean_up=result.get("current_mean",0)>=result.get("reference_mean",0); upper_up=result.get("current_quantiles",{}).get("0.95",0)>=result.get("reference_quantiles",{}).get("0.95",0)
 if not valid: reasons.append("data contract, finiteness, or non-negative count check failed")
 if missing=="critical": reasons.append("missingness increase is critical")
 if not mean_up: reasons.append("current mean decreased")
 if not upper_up: reasons.append("current 95th percentile decreased")
 # Mean and upper-tail growth are the transparent exposure test; a small median
 # movement alone is not unexpected across different chronological populations.
 expected=kind=="cumulative_history" and valid and missing!="critical" and mean_up and upper_up
 operational=raw
 if kind=="cumulative_history" and raw=="critical" and missing!="critical" and expected: operational="warning"
 reason="Expected temporal accumulation: valid cumulative counts have higher mean and 95th-percentile exposure in the later population." if expected else ("; ".join(reasons) or "Raw statistical drift is operationally retained.")
 result.update({"feature_semantics":kind,"accumulation_check_passed":expected,"accumulation_check_reasons":reasons if reasons else ["mean and 95th percentile are non-decreasing; median is reported but not used as a population-composition gate"],"raw_statistical_severity":raw,"operational_severity":operational,"operational_severity_reason":reason,"severity":operational,"expected_temporal_accumulation":expected,"temporal_shift_reason":reason if expected else None})
 return operational
def probs(values,cats,eps):
 counts=values.value_counts(dropna=False).reindex(cats,fill_value=0).astype(float); raw=counts/len(values) if len(values) else counts
 return (raw+eps)/(raw+eps).sum()
def psi(expected,actual,eps=1e-6):
 return float(((actual-expected)*np.log((actual+eps)/(expected+eps))).sum())
def numeric_drift(reference,current,bins,eps):
 r=pd.to_numeric(reference,errors="coerce"); c=pd.to_numeric(current,errors="coerce"); rv,cv=r.dropna().to_numpy(),c.dropna().to_numpy()
 if not len(rv) or not len(cv): raise MonitoringError("Numeric drift requires non-empty finite populations.")
 edges=np.unique(np.quantile(rv,np.linspace(0,1,bins+1)))
 if len(edges)<2: edges=np.array([rv.min()-1,rv.max()+1])
 else: edges[0],edges[-1]=-np.inf,np.inf
 rbin=pd.cut(r,edges,include_lowest=True).astype(str).where(r.notna(),"<missing>"); cbin=pd.cut(c,edges,include_lowest=True).astype(str).where(c.notna(),"<missing>")
 cats=sorted(set(rbin)|set(cbin)); ep,ap=probs(rbin,cats,eps),probs(cbin,cats,eps)
 return {"psi":psi(ep,ap,eps),"ks_statistic":float(ks_2samp(rv,cv).statistic),"wasserstein_distance":float(wasserstein_distance(rv,cv)),"reference_mean":float(rv.mean()),"current_mean":float(cv.mean()),"reference_median":float(np.median(rv)),"current_median":float(np.median(cv)),"reference_minimum":float(rv.min()),"current_minimum":float(cv.min()),"data_contract_valid":True,"values_finite":bool(np.isfinite(rv).all() and np.isfinite(cv).all()),"reference_quantiles":{str(q):float(np.quantile(rv,q)) for q in (.05,.5,.95)},"current_quantiles":{str(q):float(np.quantile(cv,q)) for q in (.05,.5,.95)},"missingness_change_pp":float((c.isna().mean()-r.isna().mean())*100),"bins":[{"bin":str(x),"expected_proportion":float(ep.loc[x]),"actual_proportion":float(ap.loc[x])} for x in cats]}
def categorical_drift(reference,current,eps):
 r=reference.fillna("<missing>").astype(str); c=current.fillna("<missing>").astype(str); cats=sorted(set(r)|set(c)); ep,ap=probs(r,cats,eps),probs(c,cats,eps); mid=(ep+ap)/2
 jsd=float(.5*((ep*np.log(ep/mid)).sum()+(ap*np.log(ap/mid)).sum()))
 unseen=~c.isin(set(r)); shares={x:{"reference":float(ep.loc[x]),"current":float(ap.loc[x])} for x in cats}
 return {"psi":psi(ep,ap,eps),"jensen_shannon_divergence":jsd,"categories":shares,"unseen_category_count":int(len(set(c[unseen]))),"unseen_category_row_percentage":float(unseen.mean()*100),"largest_absolute_share_change":float((ap-ep).abs().max()),"missingness_change_pp":float((current.isna().mean()-reference.isna().mean())*100)}
def quality(frame):
 return {"row_count":len(frame),"column_count":len(frame.columns),"duplicate_activity_id_count":int(frame.activity_nr.duplicated().sum()),"date_range":{"minimum":str(pd.to_datetime(frame.open_date).min().date()),"maximum":str(pd.to_datetime(frame.open_date).max().date())},"missingness":{x:float(frame[x].isna().mean()*100) for x in MODEL_COLUMNS},"numeric_finite":{x:bool(np.isfinite(pd.to_numeric(frame[x],errors="coerce").dropna()).all()) for x in NUMERIC}}
def exposure(frame,cfg):
 out={}; suppressed=0
 for feature in GROUPS:
  rows=[]
  for value,g in frame.groupby(feature,dropna=False):
   count=len(g); share=count/len(frame); row={"group":str(value),"row_count":count,"candidate_share":share}
   for label,col in (("top_5","top_5_percent_flag"),("top_10","top_10_percent_flag"),("top_20","top_20_percent_flag")):
    selected=int(g[col].sum()); total=int(frame[col].sum()); row[f"{label}_row_count"]=selected;row[f"{label}_selection_rate"]=selected/count;row[f"{label}_tier_share"]=selected/total if total else 0
   row["top_10_representation_ratio"]=None if count<cfg["minimum_group_size"] else row["top_10_tier_share"]/share if share else None
   row["suppressed_small_group"]=count<cfg["minimum_group_size"]; suppressed+=row["suppressed_small_group"];rows.append(row)
  out[feature]=rows
 return out,suppressed
def run(config_path=Path("config/monitoring_config.yaml")):
 started=time.perf_counter(); cfg=config(config_path)
 reports={k:read(Path(v)) for k,v in {"feature":"reports/feature_engineering_report.json","model":"reports/model_comparison_report.json","calibration":"reports/calibration_report.json","batch":"reports/batch_prediction_report.json","dashboard":"reports/dashboard_validation_report.json"}.items()}; foundation=read(Path("reports/data_foundation_report.json"))
 if any(x.get("status")!="PASS" for x in reports.values()):raise MonitoringError("Required reports must have status PASS.")
 snapshot,version=reports["batch"]["source_snapshot_id"],reports["batch"]["feature_version"]
 if foundation.get("snapshot_id")!=snapshot:raise MonitoringError("Data foundation snapshot is incompatible.")
 if any((x.get("source_snapshot_id"),x.get("feature_version"))!=(snapshot,version) for x in (reports["feature"],reports["model"],reports["calibration"],reports["dashboard"])):raise MonitoringError("Snapshot or feature version mismatch.")
 batch=reports["batch"]; ranked_path=Path(batch["ranked_output_path"]); manifest_path=ranked_path.parent/"prediction_manifest.json"; manifest=read(manifest_path)
 if manifest.get("prediction_run_id")!=batch["prediction_run_id"] or digest(ranked_path)!=batch["ranked_output_hash"]:raise MonitoringError("Prediction report and manifest disagree.")
 ranked=pd.read_csv(ranked_path)
 if len(ranked)!=300 or ranked.activity_nr.isna().any() or not ranked.activity_nr.is_unique:raise MonitoringError("Current candidate activity ID contract is invalid.")
 if TARGET in ranked or PROHIBITED_COLUMNS&set(ranked.columns) or not set(MODEL_COLUMNS+["raw_risk_score"]).issubset(ranked):raise MonitoringError("Current candidate schema includes prohibited or missing fields.")
 if not np.isfinite(ranked.raw_risk_score).all() or ((ranked.raw_risk_score<0)|(ranked.raw_risk_score>1)).any():raise MonitoringError("Current score range is invalid.")
 paths=reports["feature"]["output_paths"]; fmanifest=read(Path(paths["manifest"]));
 if any(fmanifest.get("file_hashes",{}).get(Path(paths[x]).name)!=digest(Path(paths[x])) for x in ("train","validation")):raise MonitoringError("Reference feature hash does not match feature manifest.")
 train,validation=pd.read_csv(paths["train"]),pd.read_csv(paths["validation"])
 for frame in (train,validation):
  if TARGET not in frame:raise MonitoringError("Reference feature target is missing before safe removal.")
  frame.drop(columns=[TARGET],inplace=True)
 for frame in (train,validation,ranked):
  if not set(MODEL_COLUMNS).issubset(frame) or PROHIBITED_COLUMNS&set(frame):raise MonitoringError("Monitored feature contract invalid.")
 reference,current=validation,ranked; secondary=train
 numeric={};categorical={}; reasons=[]
 for feature in NUMERIC:
  result=numeric_drift(reference[feature],current[feature],cfg["numeric_bins"],cfg["epsilon"]); raw=max((sev(result["psi"],cfg["psi"]["warning"],cfg["psi"]["critical"]),sev(result["missingness_change_pp"],cfg["missingness_increase_pp"]["warning"],cfg["missingness_increase_pp"]["critical"])),key=("stable","warning","critical").index);operational_severity(feature,result,raw,cfg);numeric[feature]=result
 for feature in CATEGORICAL:
  result=categorical_drift(reference[feature],current[feature],cfg["epsilon"]); raw=max((sev(result["psi"],cfg["psi"]["warning"],cfg["psi"]["critical"]),sev(result["jensen_shannon_divergence"],cfg["jsd"]["warning"],cfg["jsd"]["critical"]),sev(result["unseen_category_row_percentage"],cfg["unseen_category_percentage"]["warning"],cfg["unseen_category_percentage"]["critical"])),key=("stable","warning","critical").index);operational_severity(feature,result,raw,cfg);categorical[feature]=result
 exp=next(x for x in reports["model"]["experiments"] if x["experiment_id"]==batch["selected_day3_experiment"]); scores=pd.read_csv(Path(exp["artifact_path"]).parent/"validation_scoring.csv",usecols=["raw_probability"])["raw_probability"]
 score=numeric_drift(scores,current.raw_risk_score,cfg["numeric_bins"],cfg["epsilon"]);raw=sev(score["psi"],cfg["psi"]["warning"],cfg["psi"]["critical"]);operational_severity("raw_risk_score",score,raw,cfg);score["percentage_above_reference_median"]=float((current.raw_risk_score>scores.median()).mean()*100);score["percentage_above_reference_quantiles"]={str(q):float((current.raw_risk_score>scores.quantile(q)).mean()*100) for q in (.05,.5,.95)}
 exposure_data,suppressed=exposure(current,cfg); all_results={**numeric,**categorical,"raw_risk_score":score}; levels=[x["operational_severity"] for x in all_results.values()]; raw_levels=[x["raw_statistical_severity"] for x in all_results.values()]; health="CRITICAL" if "critical" in levels else "WARNING" if "warning" in levels else "HEALTHY"; counts={x:levels.count(x) for x in ("stable","warning","critical")}; raw_counts={x:raw_levels.count(x) for x in ("stable","warning","critical")}; expected_count=sum(x["expected_temporal_accumulation"] for x in all_results.values()); top=sorted(all_results,key=lambda x:all_results[x].get("psi",0),reverse=True)[:5]
 runid=hashlib.sha256(json.dumps({"reference":digest(Path(paths["validation"])),"current":digest(ranked_path),"prediction_manifest":digest(manifest_path),"config":digest(config_path),"snapshot":snapshot,"version":version},sort_keys=True).encode()).hexdigest()[:20]; root=Path(cfg["output_root"])/runid
 def valid():
  try:
   m=read(root/"monitoring_manifest.json");return m.get("monitoring_run_id")==runid and all(digest(root/n)==m["output_hashes"][n] for n in ("review_queue_template.csv","future_outcome_template.csv"))
  except Exception:return False
 reused=valid()
 if not reused:
  if root.exists():root.rename(root.with_name(root.name+".invalid-"+uuid.uuid4().hex[:8]))
  tmp=root.with_name(root.name+".tmp-"+uuid.uuid4().hex);tmp.mkdir(parents=True)
  try:
   review_template(current,batch["prediction_run_id"]).to_csv(tmp/"review_queue_template.csv",index=False,lineterminator="\n");future_outcome_template().to_csv(tmp/"future_outcome_template.csv",index=False,lineterminator="\n")
   m={"monitoring_run_id":runid,"source_snapshot_id":snapshot,"feature_version":version,"prediction_run_id":batch["prediction_run_id"],"output_hashes":{n:digest(tmp/n) for n in ("review_queue_template.csv","future_outcome_template.csv")}};atomic_json(tmp/"monitoring_manifest.json",m);tmp.replace(root)
  except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
 return {"status":"PASS","monitoring_health":health,"source_snapshot_id":snapshot,"feature_version":version,"prediction_run_id":batch["prediction_run_id"],"monitoring_run_id":runid,"primary_reference":"2022 validation feature population (target removed)","secondary_reference":"2020-2021 training feature population (target removed)","current_population":"2023 unlabelled ranked candidate population","reference_paths":{"validation":paths["validation"],"training":paths["train"]},"reference_hashes":{"validation":digest(Path(paths["validation"])),"training":digest(Path(paths["train"]))},"current_path":str(ranked_path),"current_hash":digest(ranked_path),"reference_current_row_counts":{"validation":len(reference),"training":len(secondary),"current":len(current)},"data_quality":{"validation":quality(reference),"current":quality(current),"schema_differences":{"missing_from_current":sorted(set(MODEL_COLUMNS)-set(current)),"extra_current_columns":sorted(set(current)-set(MODEL_COLUMNS)-{"rank","activity_nr","open_date","raw_risk_score","score_percentile","review_priority","top_5_percent_flag","top_10_percent_flag","top_20_percent_flag"})}},"numeric_drift":numeric,"categorical_drift":categorical,"score_distribution_drift":score,"severity_thresholds":cfg,"stable_feature_count":counts["stable"],"warning_feature_count":counts["warning"],"critical_feature_count":counts["critical"],"top_drifted_features":top,"overall_health_reasons":[f"{x}: {all_results[x]['severity']}" for x in top if all_results[x]["severity"]!="stable"],"review_exposure_diagnostics":exposure_data,"suppressed_small_group_count":suppressed,"review_worksheet_path":str(root/"review_queue_template.csv"),"review_worksheet_hash":digest(root/"review_queue_template.csv"),"future_outcome_template_path":str(root/"future_outcome_template.csv"),"future_outcome_template_hash":digest(root/"future_outcome_template.csv"),"monitoring_manifest_path":str(root/"monitoring_manifest.json"),"artifacts_reused":reused,"current_labels_accessed":False,"current_performance_metrics_calculated":False,"outcome_fairness_metrics_calculated":False,"model_refit_attempted":False,"prediction_artifact_modified":False,"automatic_enforcement":False,"runtime_seconds":round(time.perf_counter()-started,4),"limitations":["The 2023 candidate population has no outcomes available to this workflow.","Drift does not prove predictive performance degradation; score drift is not performance drift.","Review-exposure differences do not establish unfairness or discrimination.","Small groups are suppressed; raw scores are uncalibrated; human review is required and no enforcement is automatic.","Future complete labels are required for out-of-time performance evaluation."]}
