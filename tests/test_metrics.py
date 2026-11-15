"""Tests for metrics module."""

import numpy as np
import pandas as pd
import pytest

from segmentae.core.exceptions import ValidationError
from segmentae.metrics.performance_metrics import (
    metrics_classification,
    metrics_regression,
)


class TestMetricsClassification:
    def test_perfect_predictions(self):
        y_true = pd.Series([0, 0, 1, 1, 0])
        y_pred = pd.Series([0, 0, 1, 1, 0])
        result = metrics_classification(y_true, y_pred)

        assert isinstance(result, pd.DataFrame)
        assert result["Accuracy"].iloc[0] == 1.0
        assert result["Precision"].iloc[0] == 1.0
        assert result["Recall"].iloc[0] == 1.0
        assert result["F1 Score"].iloc[0] == 1.0

    def test_all_wrong_predictions(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = pd.Series([1, 1, 0, 0])
        result = metrics_classification(y_true, y_pred)
        assert result["Accuracy"].iloc[0] == 0.0

    def test_with_numpy_arrays(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        result = metrics_classification(y_true, y_pred)
        assert "Accuracy" in result.columns

    def test_length_mismatch_raises(self):
        y_true = pd.Series([0, 1, 1])
        y_pred = pd.Series([0, 1])
        with pytest.raises(ValidationError, match="Length mismatch"):
            metrics_classification(y_true, y_pred)

    def test_empty_arrays_raises(self):
        with pytest.raises(ValidationError, match="Empty"):
            metrics_classification(pd.Series([], dtype=int), pd.Series([], dtype=int))

    def test_returns_dataframe_columns(self):
        result = metrics_classification(pd.Series([0, 1]), pd.Series([0, 1]))
        expected_cols = {"Accuracy", "Precision", "Recall", "F1 Score"}
        assert set(result.columns) == expected_cols


class TestMetricsRegression:
    def test_perfect_predictions(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0, 3.0])
        result = metrics_regression(y_true, y_pred)

        assert isinstance(result, pd.DataFrame)
        assert result["Mean Squared Error"].iloc[0] == 0.0
        assert result["Mean Absolute Error"].iloc[0] == 0.0
        assert result["R-squared"].iloc[0] == 1.0

    def test_with_numpy_arrays(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = metrics_regression(y_true, y_pred)
        assert result["Mean Absolute Error"].iloc[0] == pytest.approx(0.1, abs=1e-6)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="Length mismatch"):
            metrics_regression(pd.Series([1.0]), pd.Series([1.0, 2.0]))

    def test_empty_arrays_raises(self):
        with pytest.raises(ValidationError, match="Empty"):
            metrics_regression(pd.Series([], dtype=float), pd.Series([], dtype=float))

    def test_returns_dataframe_columns(self):
        result = metrics_regression(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
        expected_cols = {
            "Mean Absolute Error",
            "Mean Squared Error",
            "Root Mean Squared Error",
            "R-squared",
            "Max Error",
        }
        assert set(result.columns) == expected_cols
