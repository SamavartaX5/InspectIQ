import unittest
import tempfile
from pathlib import Path
import json
from src.calibration import select_method

CFG={"minimum_brier_improvement":0.0001,"recall_at_10_tolerance":0.02}
def metrics(brier,recall=.23,logloss=.5,ece=.04): return {"brier_score":brier,"log_loss":logloss,"expected_calibration_error":ece,"ranking_at":{"10":{"recall":recall}}}
class CalibrationSelectionTests(unittest.TestCase):
 def test_uncalibrated_wins_when_calibrators_worsen_brier(self):
  selected,improved,_=select_method({"uncalibrated":metrics(.17),"sigmoid":metrics(.18),"isotonic":metrics(.175)},CFG);self.assertEqual((selected,improved),("uncalibrated",False))
 def test_ece_cannot_override_brier(self):
  selected,_,_=select_method({"uncalibrated":metrics(.17,ece=.05),"sigmoid":metrics(.18,ece=.001),"isotonic":metrics(.176,ece=.001)},CFG);self.assertEqual(selected,"uncalibrated")
 def test_sigmoid_wins_when_meaningfully_better(self):
  selected,improved,_=select_method({"uncalibrated":metrics(.17),"sigmoid":metrics(.16),"isotonic":metrics(.165)},CFG);self.assertEqual((selected,improved),("sigmoid",True))
 def test_isotonic_wins_when_meaningfully_better(self):
  selected,improved,_=select_method({"uncalibrated":metrics(.17),"sigmoid":metrics(.165),"isotonic":metrics(.16)},CFG);self.assertEqual((selected,improved),("isotonic",True))
 def test_recall_tolerance_is_enforced(self):
  selected,improved,_=select_method({"uncalibrated":metrics(.17,.23),"sigmoid":metrics(.16,.20),"isotonic":metrics(.18,.23)},CFG);self.assertEqual((selected,improved),("uncalibrated",False))
 def test_uncalibrated_report_uses_generic_existing_final_artifact(self):
  with tempfile.TemporaryDirectory() as directory:
   artifact=Path(directory)/"final_candidate.joblib";artifact.write_bytes(b"fixture artifact")
   report={"selected_calibration_method":"uncalibrated","final_package_type":"uncalibrated","final_calibration_applied":False,"final_package_calibration_period":"not applicable","final_candidate_artifact_path":str(artifact),"locked_test_labels_accessed":False,"locked_test_metrics_calculated":False,"locked_test_predictions_created":False}
   self.assertEqual(report["selected_calibration_method"],"uncalibrated")
   self.assertEqual(report["final_package_type"],"uncalibrated")
   self.assertFalse(report["final_calibration_applied"])
   self.assertEqual(report["final_package_calibration_period"],"not applicable")
   self.assertNotIn("final_calibrated_artifact_path",report)
   self.assertTrue(Path(report["final_candidate_artifact_path"]).exists())
   self.assertFalse(report["locked_test_labels_accessed"] or report["locked_test_metrics_calculated"] or report["locked_test_predictions_created"])
