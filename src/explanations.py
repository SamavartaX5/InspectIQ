"""Read-only, deterministic explanations for the fitted InspectIQ candidate model."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.batch_prediction import MODEL_COLUMNS, expected_model_columns, positive_class_index


FEATURE_LABELS = {
    "naics_group": "NAICS group", "insp_type": "Inspection type", "insp_scope": "Inspection scope",
    "owner_type": "Owner type", "safety_hlth": "Safety/health indicator",
    "nr_in_estab": "Reported establishment size", "open_month": "Inspection month",
    "industry_prior_inspection_count": "Prior industry inspections",
    "industry_prior_positive_count": "Prior industry positive findings",
    "industry_prior_positive_rate_smoothed": "Smoothed prior industry positive rate",
    "industry_history_status": "Industry history status",
}
NUMERIC_FEATURES = {
    "nr_in_estab", "open_month", "industry_prior_inspection_count", "industry_prior_positive_count",
    "industry_prior_positive_rate_smoothed",
}


class ExplanationError(RuntimeError):
    pass


def training_references(training: pd.DataFrame) -> dict[str, Any]:
    """Return medians/modes using only historic training features."""
    if not set(MODEL_COLUMNS).issubset(training.columns):
        raise ExplanationError("Training feature artifact lacks required model columns.")
    references: dict[str, Any] = {}
    for feature in MODEL_COLUMNS:
        values = training[feature]
        if feature in NUMERIC_FEATURES:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            if numeric.empty:
                raise ExplanationError(f"Training reference cannot be calculated for {feature}.")
            references[feature] = float(numeric.median())
        else:
            text = values.fillna("<missing>").astype(str)
            counts = text.value_counts()
            maximum = counts.max()
            references[feature] = sorted(counts[counts == maximum].index.tolist())[0]
    return references


def _positive_score(model: Any, frame: pd.DataFrame) -> float:
    if not callable(getattr(model, "predict_proba", None)):
        raise ExplanationError("Model does not implement predict_proba.")
    columns = expected_model_columns(model)
    probabilities = np.asarray(model.predict_proba(frame.loc[:, columns]), dtype=float)
    index = positive_class_index(model)
    if probabilities.shape != (1, len(getattr(model, "classes_", getattr(getattr(model, "named_steps", {}).get("model"), "classes_", [])))):
        # Pipelines commonly expose classes_ at the top level; allow the model-step fallback.
        if probabilities.ndim != 2 or probabilities.shape[0] != 1 or index >= probabilities.shape[1]:
            raise ExplanationError("Model returned an invalid probability matrix.")
    score = float(probabilities[0, index])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ExplanationError("Model returned an invalid raw risk score.")
    return score


def local_perturbation_explanation(model: Any, candidate: pd.Series, references: dict[str, Any]) -> pd.DataFrame:
    """Measure one-at-a-time score changes; this is not a causal explanation."""
    if not set(MODEL_COLUMNS).issubset(candidate.index) or not set(MODEL_COLUMNS).issubset(references):
        raise ExplanationError("Candidate or training reference feature contract is incomplete.")
    original = pd.DataFrame([candidate.loc[MODEL_COLUMNS].to_dict()])
    original_score = _positive_score(model, original)
    rows = []
    for feature in MODEL_COLUMNS:
        perturbed = original.copy()
        if feature not in NUMERIC_FEATURES:
            perturbed[feature] = perturbed[feature].astype("object")
        perturbed.loc[0, feature] = references[feature]
        perturbed_score = _positive_score(model, perturbed)
        rows.append({
            "feature": feature, "feature_label": FEATURE_LABELS[feature],
            "observed_value": candidate[feature], "reference_value": references[feature],
            "raw_score_difference": original_score - perturbed_score,
            "direction": "increased_score" if original_score > perturbed_score else "decreased_score" if original_score < perturbed_score else "little_sensitivity",
        })
    result = pd.DataFrame(rows).sort_values(["raw_score_difference", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    result.attrs["caveat"] = "One-feature-at-a-time local score sensitivity using training-only references; it is neither SHAP nor causal attribution."
    result.attrs["original_raw_risk_score"] = original_score
    return result


def global_feature_importance(model: Any) -> pd.DataFrame:
    """Aggregate fitted Random Forest transformed importances by source feature."""
    try:
        preprocess = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        transformed = [str(value) for value in preprocess.get_feature_names_out()]
        values = np.asarray(estimator.feature_importances_, dtype=float)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ExplanationError("Final model does not expose a defensible fitted Random Forest importance contract.") from exc
    if len(transformed) != len(values) or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ExplanationError("Final model transformed feature importances are invalid.")
    aggregate = {feature: 0.0 for feature in MODEL_COLUMNS}
    for transformed_name, value in zip(transformed, values):
        tail = transformed_name.split("__", 1)[-1]
        matches = [feature for feature in MODEL_COLUMNS if tail == feature or tail.startswith(feature + "_")]
        if not matches:
            raise ExplanationError(f"Cannot map transformed feature name to a source feature: {transformed_name}")
        aggregate[max(matches, key=len)] += float(value)
    total = sum(aggregate.values())
    if total <= 0:
        raise ExplanationError("Final model feature importances have no positive mass.")
    rows = [{"feature": feature, "feature_label": FEATURE_LABELS[feature], "importance": value / total} for feature, value in aggregate.items()]
    return pd.DataFrame(rows).sort_values(["importance", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
