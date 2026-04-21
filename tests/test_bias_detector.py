"""Unit tests for bias/metrics.py and bias/detector.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from rai_toolkit.bias.detector import BiasDetector, BiasReport, _compute_risk_level
from rai_toolkit.bias.metrics import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def biased_dataset():
    """200-sample dataset where females are approved at half the rate of males."""
    np.random.seed(0)
    n = 200
    gender = np.array(["male"] * 120 + ["female"] * 80)
    # Males: 70% approved, females: 20% approved
    y_true = np.array([1] * 84 + [0] * 36 + [1] * 16 + [0] * 64)
    y_pred = y_true.copy()  # perfect predictor to isolate metric math
    return gender, y_true, y_pred


@pytest.fixture()
def sklearn_dataset():
    """Small DataFrame + fitted LogisticRegression for end-to-end tests."""
    np.random.seed(42)
    n = 300
    gender = np.array(["male"] * 200 + ["female"] * 100)
    credit = np.concatenate([np.random.randint(620, 800, 200), np.random.randint(550, 720, 100)])
    income = np.concatenate([np.random.randint(40000, 90000, 200), np.random.randint(25000, 70000, 100)])

    # Bias: flip 80% of female approvals
    y_true = (credit > 680).astype(int)
    female_ok = np.where((gender == "female") & (y_true == 1))[0]
    flipped = np.random.choice(female_ok, size=int(len(female_ok) * 0.8), replace=False)
    y_biased = y_true.copy()
    y_biased[flipped] = 0

    X = pd.DataFrame({"credit_score": credit, "income": income, "gender": gender})
    model = LogisticRegression(random_state=0).fit(X[["credit_score", "income"]], y_biased)
    return model, X, y_true


# ---------------------------------------------------------------------------
# metrics.py — pure functions
# ---------------------------------------------------------------------------

class TestDemographicParityDifference:
    def test_perfect_parity(self):
        # Both groups selected at 50% -> difference should be 0
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        assert demographic_parity_difference(y_true, y_pred, groups) == pytest.approx(0.0)

    def test_known_values(self):
        # Group A: 3/4 selected (0.75), Group B: 1/4 selected (0.25) -> diff = 0.50
        y_true = np.ones(8)
        y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        result = demographic_parity_difference(y_true, y_pred, groups)
        assert result == pytest.approx(0.50, abs=1e-6)

    def test_returns_float(self, biased_dataset):
        gender, y_true, y_pred = biased_dataset
        result = demographic_parity_difference(y_true, y_pred, gender)
        assert isinstance(result, float)
        assert result >= 0


class TestDisparateImpactRatio:
    def test_perfect_parity(self):
        y_true = np.ones(8)
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        assert disparate_impact_ratio(y_true, y_pred, groups) == pytest.approx(1.0)

    def test_known_ratio(self):
        # Group A: 3/4 = 0.75, Group B: 1/4 = 0.25 -> ratio = 0.25/0.75 = 0.333
        y_true = np.ones(8)
        y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        result = disparate_impact_ratio(y_true, y_pred, groups)
        assert result == pytest.approx(1 / 3, abs=1e-6)

    def test_zero_max_rate_returns_zero(self):
        # No one gets approved -> max_rate=0, ratio should be 0 not ZeroDivisionError
        y_true = np.ones(4)
        y_pred = np.zeros(4)
        groups = np.array(["A", "A", "B", "B"])
        assert disparate_impact_ratio(y_true, y_pred, groups) == 0.0

    def test_below_four_fifths_rule(self, biased_dataset):
        gender, y_true, y_pred = biased_dataset
        # females approved at 20%, males at 70% -> ratio ≈ 0.286 (below 0.8)
        result = disparate_impact_ratio(y_true, y_pred, gender)
        assert result < 0.8


# ---------------------------------------------------------------------------
# detector.py — risk level + BiasDetector
# ---------------------------------------------------------------------------

class TestComputeRiskLevel:
    def test_low(self):
        assert _compute_risk_level(dpd=0.05, dir_=0.90) == "LOW"

    def test_medium_from_dpd(self):
        assert _compute_risk_level(dpd=0.15, dir_=0.85) == "MEDIUM"

    def test_medium_from_dir(self):
        assert _compute_risk_level(dpd=0.05, dir_=0.75) == "MEDIUM"

    def test_high_from_dpd(self):
        assert _compute_risk_level(dpd=0.25, dir_=0.85) == "HIGH"

    def test_high_from_dir(self):
        assert _compute_risk_level(dpd=0.05, dir_=0.55) == "HIGH"

    def test_high_wins_over_medium(self):
        # dir triggers HIGH even if dpd would only be MEDIUM
        assert _compute_risk_level(dpd=0.15, dir_=0.50) == "HIGH"


class TestBiasDetectorAnalyze:
    def test_returns_bias_report(self, sklearn_dataset):
        model, X, y_true = sklearn_dataset
        report = BiasDetector().analyze(model, X, y_true, sensitive_col="gender")
        assert isinstance(report, BiasReport)

    def test_risk_level_high_on_biased_data(self, sklearn_dataset):
        model, X, y_true = sklearn_dataset
        report = BiasDetector().analyze(model, X, y_true, sensitive_col="gender")
        assert report.risk_level == "HIGH"

    def test_groups_populated(self, sklearn_dataset):
        model, X, y_true = sklearn_dataset
        report = BiasDetector().analyze(model, X, y_true, sensitive_col="gender")
        assert set(report.groups) == {"male", "female"}

    def test_recommendations_not_empty(self, sklearn_dataset):
        model, X, y_true = sklearn_dataset
        report = BiasDetector().analyze(model, X, y_true, sensitive_col="gender")
        assert len(report.recommendations) > 0

    def test_missing_sensitive_col_raises(self, sklearn_dataset):
        model, X, y_true = sklearn_dataset
        with pytest.raises(ValueError, match="not found in X"):
            BiasDetector().analyze(model, X, y_true, sensitive_col="nonexistent")

    def test_low_risk_on_fair_model(self):
        # DummyClassifier predicts the most frequent class — no group disparity
        np.random.seed(1)
        n = 200
        groups = np.array(["A"] * 100 + ["B"] * 100)
        y = np.array([1] * 100 + [0] * 100)
        X = pd.DataFrame({"feature": np.random.rand(n), "group": groups})
        model = DummyClassifier(strategy="most_frequent").fit(X[["feature"]], y)
        report = BiasDetector().analyze(model, X, y, sensitive_col="group")
        # DummyClassifier predicts same class for everyone -> DPD=0
        assert report.demographic_parity_diff == pytest.approx(0.0, abs=1e-6)
