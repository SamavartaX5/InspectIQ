from __future__ import annotations
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss,log_loss,precision_score,recall_score,f1_score,confusion_matrix
from src.training import evaluate
class CalibrationError(RuntimeError): pass
def select_method(results,config):
 unc=results["uncalibrated"]
 qualified=[name for name in ("sigmoid","isotonic") if results[name]["brier_score"] < unc["brier_score"]-config["minimum_brier_improvement"] and results[name]["ranking_at"]["10"]["recall"] >= unc["ranking_at"]["10"]["recall"]-config["recall_at_10_tolerance"]]
 if not qualified:
  return "uncalibrated",False,"Calibration was evaluated but rejected because neither calibrated method meaningfully improved validation Brier score while preserving Recall@10 tolerance."
 chosen=min(qualified,key=lambda name:(results[name]["brier_score"],results[name]["log_loss"],results[name]["expected_calibration_error"],0 if name=="sigmoid" else 1))
 return chosen,True,"A calibrated method qualified by meaningful Brier improvement and Recall@10 tolerance; selected by Brier, log loss, ECE, then sigmoid tie preference."
def bins(prob,y,n):
 out=[]; ece=0.; mce=0.
 for i in range(n):
  lo=i/n; hi=(i+1)/n; mask=(prob>=lo)&((prob<hi) if i<n-1 else (prob<=hi)); c=int(mask.sum())
  mean=float(prob[mask].mean()) if c else None; obs=float(y[mask].mean()) if c else None; gap=abs(mean-obs) if c else 0.; ece+=c/len(y)*gap; mce=max(mce,gap); out.append({"lower_bound":lo,"upper_bound":hi,"row_count":c,"mean_predicted_probability":mean,"observed_positive_rate":obs,"absolute_calibration_gap":gap})
 return out,ece,mce
def metrics(prob,y,ids,bins_count):
 if not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any(): raise CalibrationError("Invalid calibrated probabilities")
 ranked,rank=evaluate(prob,y,ids); b,e,m=bins(prob,y,bins_count); p=(prob>=.5).astype(int)
 rank.update({"brier_score":float(brier_score_loss(y,prob)),"log_loss":float(log_loss(y,prob,labels=[0,1])),"expected_calibration_error":e,"maximum_calibration_error":m,"mean_predicted_probability":float(prob.mean()),"observed_positive_rate":float(y.mean()),"mean_probability_gap":abs(float(prob.mean()-y.mean())),"probability_range":{"min":float(prob.min()),"max":float(prob.max()),"finite":True},"reliability_bins":b,"threshold_0_5":{"precision":float(precision_score(y,p,zero_division=0)),"recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),"confusion_matrix":confusion_matrix(y,p,labels=[0,1]).tolist()}})
 return ranked,rank
def calibrate(base,x,y,method):
 return CalibratedClassifierCV(FrozenEstimator(base),method=method).fit(x,y)
