"""BiasDetector: runs fairness analysis on a Scikit-learn model."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from loguru import logger

from rai_toolkit.bias.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    disparate_impact_ratio,
)

# Risk thresholds (documented here as the single source of truth)
# Demographic parity difference: >0.1 → MEDIUM, >0.2 → HIGH
# Disparate impact ratio:        <0.8 → MEDIUM, <0.6 → HIGH
_DPD_MEDIUM = 0.1
_DPD_HIGH = 0.2
_DIR_MEDIUM = 0.8
_DIR_LOW_BOUND = 0.6  # below this → HIGH

_RECOMMENDATIONS: dict[Literal["LOW", "MEDIUM", "HIGH"], list[str]] = {
    "LOW": [
        "Le modèle présente une équité acceptable sur cette variable sensible.",
        "Continuer à surveiller les métriques à chaque réentraînement.",
    ],
    "MEDIUM": [
        "Appliquer une contrainte d'équité Fairlearn (ExponentiatedGradient) lors du réentraînement.",
        "Analyser les features corrélées à la variable sensible (proxy discrimination).",
        "Documenter les écarts dans la model card avant toute mise en production.",
    ],
    "HIGH": [
        "Ne pas déployer ce modèle sans correction préalable.",
        "Appliquer un post-processing de rééquilibrage (ThresholdOptimizer de Fairlearn).",
        "Effectuer un audit humain des décisions impactées par ce modèle.",
        "Notifier le responsable conformité — seuil réglementaire dépassé (règle des 4/5).",
    ],
}


class BiasReport(BaseModel):
    """Result of a bias analysis for one sensitive attribute."""

    sensitive_column: str
    groups: list[str]
    demographic_parity_diff: float
    equal_opportunity_diff: float
    disparate_impact_ratio: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommendations: list[str]


def _compute_risk_level(dpd: float, dir_: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    """Derive the overall risk level from the two key metrics."""
    level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"

    if dpd > _DPD_HIGH or dir_ < _DIR_LOW_BOUND:
        level = "HIGH"
    elif dpd > _DPD_MEDIUM or dir_ < _DIR_MEDIUM:
        level = "MEDIUM"

    return level


class BiasDetector:
    """Runs fairness metrics on a fitted Scikit-learn model.

    Usage:
        detector = BiasDetector()
        report = detector.analyze(model, X, y_true, sensitive_col="gender")
    """

    def analyze(
        self,
        model,
        X: pd.DataFrame,
        y_true: np.ndarray,
        sensitive_col: str,
    ) -> BiasReport:
        """Compute fairness metrics for one sensitive attribute.

        Args:
            model: Any fitted Scikit-learn estimator with a predict() method.
            X: Feature matrix including the sensitive column.
            y_true: Ground-truth binary labels (0/1).
            sensitive_col: Name of the column in X to use as the sensitive feature.

        Returns:
            BiasReport with metrics, risk level, and recommendations.
        """
        if sensitive_col not in X.columns:
            raise ValueError(
                f"Column '{sensitive_col}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        sensitive_features = X[sensitive_col].to_numpy()
        groups = sorted(X[sensitive_col].unique().tolist())

        logger.info(
            f"Running bias analysis on '{sensitive_col}' "
            f"with groups {groups} (n={len(y_true)})"
        )

        # Use only the features the model was trained on (sklearn sets feature_names_in_)
        if hasattr(model, "feature_names_in_"):
            X_model = X[list(model.feature_names_in_)]
        else:
            X_model = X.drop(columns=[sensitive_col], errors="ignore")
        y_pred = model.predict(X_model)

        dpd = float(demographic_parity_difference(y_true, y_pred, sensitive_features))
        eod = float(equal_opportunity_difference(y_true, y_pred, sensitive_features))
        dir_ = float(disparate_impact_ratio(y_true, y_pred, sensitive_features))

        risk_level = _compute_risk_level(dpd, dir_)

        logger.info(
            f"Bias results — DPD={dpd:.3f}, EOD={eod:.3f}, "
            f"DIR={dir_:.3f} → risk={risk_level}"
        )

        return BiasReport(
            sensitive_column=sensitive_col,
            groups=[str(g) for g in groups],
            demographic_parity_diff=round(dpd, 4),
            equal_opportunity_diff=round(eod, 4),
            disparate_impact_ratio=round(dir_, 4),
            risk_level=risk_level,
            recommendations=_RECOMMENDATIONS[risk_level],
        )
