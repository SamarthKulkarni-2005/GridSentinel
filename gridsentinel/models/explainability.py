"""SHAP and LIME explainability for the TheftDetector model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    output_path: str | Path = "outputs/shap_summary.png",
) -> str:
    """Generate and save SHAP beeswarm summary plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
        )
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP summary plot saved to %s", output_path)
        return str(output_path)
    except Exception as exc:
        logger.error("Failed to generate SHAP plot: %s", exc)
        return str(output_path)


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer for XGBoost model."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classification, shap_values may be a list [neg, pos]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        logger.info("SHAP values computed: shape %s", np.array(shap_values).shape)
        return np.array(shap_values)
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        # Return zeros as fallback
        return np.zeros_like(X, dtype=float)


def explain_instance_lime(
    x: np.ndarray,
    predict_fn,
    X_train: np.ndarray,
    feature_names: List[str],
    num_features: int = 10,
) -> Dict[str, float]:
    """Generate LIME explanation for a single prediction instance."""
    try:
        from lime import lime_tabular

        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=["normal", "theft"],
            mode="classification",
            random_state=42,
        )
        explanation = explainer.explain_instance(
            x, predict_fn, num_features=num_features
        )
        # Extract feature weights
        weights = {feat: float(weight) for feat, weight in explanation.as_list()}
        logger.debug("LIME explanation computed with %d features", len(weights))
        return weights
    except Exception as exc:
        logger.error("LIME explanation failed: %s", exc)
        return {name: 0.0 for name in feature_names[:num_features]}


def get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    n: int = 10,
) -> List[Dict[str, float]]:
    """Return top-n features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0) if shap_values.ndim > 1 else np.abs(shap_values)
    idx_sorted = np.argsort(mean_abs)[::-1][:n]
    return [
        {"name": feature_names[i], "shap_value": float(mean_abs[i])}
        for i in idx_sorted
    ]
