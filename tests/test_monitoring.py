import unittest
import numpy as np
import pandas as pd
from src.governance import OUTCOME_COLUMNS, REVIEW_COLUMNS, future_outcome_template, review_template
from src.monitoring import categorical_drift, numeric_drift, psi, sev, exposure, operational_severity

class MonitoringTests(unittest.TestCase):
 def test_numeric_psi_reference_bins_missing_and_distances(self):
  result=numeric_drift(pd.Series([0.,0.,1.,1.,np.nan]),pd.Series([1.,1.,1.,1.,np.nan]),2,1e-6)
  self.assertGreater(result["psi"],0);self.assertGreater(result["ks_statistic"],0);self.assertTrue(any(x["bin"]=="<missing>" for x in result["bins"]))
  self.assertAlmostEqual(psi(pd.Series([.5,.5]),pd.Series([.5,.5])),0.0);self.assertEqual(sev(.25,.1,.25),"critical")
 def test_categorical_union_unseen_and_severity(self):
  result=categorical_drift(pd.Series(["a","a",None]),pd.Series(["a","b",None]),1e-6)
  self.assertIn("<missing>",result["categories"]);self.assertIn("b",result["categories"]);self.assertEqual(result["unseen_category_count"],1);self.assertAlmostEqual(result["unseen_category_row_percentage"],100/3)
 def test_review_exposure_and_governance_templates_are_advisory(self):
  frame=pd.DataFrame({"rank":[1,2,3],"activity_nr":["1","2","3"],"open_date":["2023-01-01"]*3,"raw_risk_score":[.9,.5,.1],"review_priority":["highest_priority","high_priority","standard_priority"],"top_5_percent_flag":[True,False,False],"top_10_percent_flag":[True,True,False],"top_20_percent_flag":[True,True,True],"naics_group":["a","a","b"],"insp_type":["x"]*3,"insp_scope":["x"]*3,"owner_type":["x"]*3,"safety_hlth":["x"]*3,"industry_history_status":["x"]*3})
  data,suppressed=exposure(frame,{"minimum_group_size":2});self.assertEqual(data["naics_group"][0]["top_10_row_count"],2);self.assertGreaterEqual(suppressed,1)
  review=review_template(frame,"run");self.assertEqual(list(review),REVIEW_COLUMNS);self.assertTrue((review["reviewer_decision"]=="").all());self.assertNotIn("automatic_enforcement",review)
  self.assertEqual(list(future_outcome_template()),OUTCOME_COLUMNS);self.assertTrue(future_outcome_template().empty)
 def test_cumulative_raw_drift_is_visible_but_expected_growth_is_operational_warning(self):
  cfg={"feature_semantics":{"industry_prior_inspection_count":"cumulative_history","raw_risk_score":"model_score"},"missingness_increase_pp":{"warning":2,"critical":5}}
  cumulative={"reference_mean":10,"current_mean":20,"reference_median":8,"current_median":15,"missingness_change_pp":0}
  self.assertEqual(operational_severity("industry_prior_inspection_count",cumulative,"critical",cfg),"warning");self.assertEqual(cumulative["raw_statistical_severity"],"critical");self.assertTrue(cumulative["expected_temporal_accumulation"])
  self.assertTrue(cumulative["accumulation_check_passed"]);self.assertTrue(cumulative["operational_severity_reason"])
  score={"reference_mean":.2,"current_mean":.3,"reference_median":.2,"current_median":.3,"missingness_change_pp":0}
  self.assertEqual(operational_severity("raw_risk_score",score,"critical",cfg),"critical")
 def test_critical_missingness_is_not_downgraded_and_dashboard_uses_safe_chart_text(self):
  cfg={"feature_semantics":{"industry_prior_positive_count":"cumulative_history"},"missingness_increase_pp":{"warning":2,"critical":5}}
  row={"reference_mean":1,"current_mean":2,"reference_median":1,"current_median":2,"missingness_change_pp":5}
  self.assertEqual(operational_severity("industry_prior_positive_count",row,"critical",cfg),"critical")
  text=__import__("pathlib").Path("app/streamlit_app.py").read_text(encoding="utf8");self.assertNotIn("use_container_width",text);self.assertIn("score_bins.index = score_bins.index.astype(str)",text)
 def test_decreasing_or_invalid_cumulative_values_remain_critical(self):
  cfg={"feature_semantics":{"any_history":"cumulative_history"},"missingness_increase_pp":{"warning":2,"critical":5}}
  decreasing={"reference_mean":10,"current_mean":9,"reference_median":8,"current_median":7,"reference_quantiles":{"0.95":20},"current_quantiles":{"0.95":19},"reference_minimum":0,"current_minimum":0,"missingness_change_pp":0}
  self.assertEqual(operational_severity("any_history",decreasing,"critical",cfg),"critical");self.assertFalse(decreasing["accumulation_check_passed"])
  invalid={"reference_mean":1,"current_mean":2,"reference_median":1,"current_median":2,"reference_quantiles":{"0.95":2},"current_quantiles":{"0.95":3},"reference_minimum":0,"current_minimum":-1,"missingness_change_pp":0}
  self.assertEqual(operational_severity("any_history",invalid,"critical",cfg),"critical")
 def test_dashboard_filters_are_queue_only(self):
  text=__import__("pathlib").Path("app/streamlit_app.py").read_text(encoding="utf8");self.assertIn('if page == "Review Queue":\n        st.sidebar.header("Review queue controls")',text)
