"""Pure fairness metric functions wrapping Fairlearn."""

import numpy as np
import fairlearn.metrics as fl
from fairlearn.metrics import MetricFrame


def demographic_parity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> float:
    """Difference in selection rate between the most and least favoured groups.

    A value of 0 means all groups are selected at the same rate.
    A positive value means some groups are selected more than others.
    """
    return fl.demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> float:
    """Difference in true positive rate (recall) between the worst and best group.

    A value of 0 means all groups have the same recall.
    A positive value means some groups are correctly identified less often.
    """
    return fl.equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )


def disparate_impact_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> float:
    """Ratio of selection rates: min(group rate) / max(group rate).

    A value of 1.0 means perfect parity. Below 0.8 is the legal threshold
    in many jurisdictions (the 4/5ths rule from US EEOC guidelines).
    Returns 0.0 if the most favoured group has a selection rate of 0.
    """
    frame = MetricFrame(
        metrics=fl.selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    rates = frame.by_group
    max_rate = rates.max()
    if max_rate == 0:
        return 0.0
    return float(rates.min() / max_rate)
