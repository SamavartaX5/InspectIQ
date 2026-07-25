"""Chronological Day 3 sklearn training and validation-only comparison."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from typing import Any
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from src.ranking_metrics import validation_metrics, selection_count

TARGET="serious_violation_found"; META=["activity_nr","open_date"]
CATEGORICAL=["naics_group","insp_type","insp_scope","owner_type","safety_hlth","industry_history_status"]
NUMERIC=["nr_in_estab","open_month","industry_prior_inspection_count","industry_prior_positive_count","industry_prior_positive_rate_smoothed"]
PROHIBITED={"citation_id","viol_type","delete_flag","current_penalty","initial_penalty","issuance_date","contest_date","final_order_date","close_case_date","close_conf_date","case_mod_date","why_no_insp"}
class TrainingError(RuntimeError): pass
def sha(path:Path)->str:
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def read_json(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding="utf-8"))
def load_inputs(manifest_path:Path)->tuple[dict[str,pd.DataFrame],dict[str,Any]]:
 m=read_json(manifest_path); root=manifest_path.parent; names={"train":"train_features.csv","validation":"validation_features.csv","test":"test_locked_features.csv"}; out={}
 for split,name in names.items():
  p=root/name
  if not p.exists() or sha(p)!=m.get("file_hashes",{}).get(name): raise TrainingError(f"Feature artifact hash failure: {p}")
  out[split]=pd.read_csv(p)
 if TARGET not in out["train"] or TARGET not in out["validation"] or TARGET in out["test"]: raise TrainingError("Feature target contract is invalid.")
 expected=set(CATEGORICAL+NUMERIC+META)
 for split,frame in out.items():
  if frame.empty or not expected.issubset(frame): raise TrainingError(f"Feature schema is invalid for {split}.")
  if PROHIBITED&set(frame): raise TrainingError("Prohibited leakage column present.")
  if not frame.activity_nr.is_unique: raise TrainingError("activity_nr values must be unique.")
 if set(out["train"].activity_nr)&set(out["validation"].activity_nr) or set(out["train"].activity_nr)&set(out["test"].activity_nr) or set(out["validation"].activity_nr)&set(out["test"].activity_nr): raise TrainingError("Split activity IDs overlap.")
 if not pd.to_datetime(out["train"].open_date).max()<pd.to_datetime(out["validation"].open_date).min()<pd.to_datetime(out["test"].open_date).min(): raise TrainingError("Split dates are not strictly chronological.")
 for split in ("train","validation"):
  values=set(out[split][TARGET]);
  if values-{0,1} or values!={0,1}: raise TrainingError(f"{split} target must be binary with both classes.")
 return out,m
def preprocessor(name:str):
 counts=["nr_in_estab","industry_prior_inspection_count","industry_prior_positive_count"]
 rest=[x for x in NUMERIC if x not in counts]
 scale=name=="logistic_regression"
 count_steps=[("impute",SimpleImputer(strategy="median")),("log1p",FunctionTransformer(np.log1p,feature_names_out="one-to-one"))]
 rest_steps=[("impute",SimpleImputer(strategy="median"))]
 if scale: count_steps.append(("scale",StandardScaler())); rest_steps.append(("scale",StandardScaler()))
 cat=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
 return ColumnTransformer([("counts",Pipeline(count_steps),counts),("numeric",Pipeline(rest_steps),rest),("categorical",cat,CATEGORICAL)],sparse_threshold=0)
def pipeline(spec):
 n=spec["model_name"]
 if n=="logistic_regression": model=LogisticRegression(C=spec["C"],class_weight=spec["class_weight"],max_iter=2000,random_state=42)
 elif n=="random_forest": model=RandomForestClassifier(n_estimators=spec["n_estimators"],max_depth=spec["max_depth"],min_samples_leaf=spec["min_samples_leaf"],class_weight=spec["class_weight"],random_state=42,n_jobs=1)
 else: model=HistGradientBoostingClassifier(learning_rate=spec["learning_rate"],max_leaf_nodes=spec["max_leaf_nodes"],max_iter=spec["max_iter"],l2_regularization=spec["l2_regularization"],random_state=42)
 return Pipeline([("preprocess",preprocessor(n)),("model",model)])
def evaluate(prob,y,ids):
 if len(prob)!=len(y) or not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any(): raise TrainingError("Invalid validation probabilities.")
 rows=[{"activity_nr":str(i),"baseline_score":float(p),"actual_label":int(a)} for i,p,a in zip(ids,prob,y)]
 rows.sort(key=lambda x:(-x["baseline_score"],int(x["activity_nr"]) if str(x["activity_nr"]).isdigit() else str(x["activity_nr"])))
 metrics=validation_metrics(rows); predicted=(prob>=.5).astype(int)
 metrics.update({"brier_score":float(brier_score_loss(y,prob)),"threshold_0_5":{"precision":float(precision_score(y,predicted,zero_division=0)),"recall":float(recall_score(y,predicted,zero_division=0)),"f1":float(f1_score(y,predicted,zero_division=0)),"confusion_matrix":confusion_matrix(y,predicted,labels=[0,1]).tolist()},"probability_range":{"min":float(prob.min()),"max":float(prob.max()),"finite":True}})
 return rows,metrics
