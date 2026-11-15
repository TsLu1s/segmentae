"""Tests for reconstruction module."""

import numpy as np
import pandas as pd
import pytest

from segmentae.core.constants import ThresholdMetric
from segmentae.pipeline.reconstruction import (
    EvaluationConfig,
    ReconstructionConfig,
    compute_column_metrics,
    compute_reconstruction_errors,
    compute_threshold,
    create_metrics_dataframe,
    detect_anomalies,
)


class TestReconstructionConfig:
    def test_default_metric(self):
        config = ReconstructionConfig()
        assert config.threshold_metric == ThresholdMetric.MSE

    def test_string_metric(self):
        config = ReconstructionConfig(threshold_metric="mae")
        assert config.threshold_metric == ThresholdMetric.MAE

    def test_enum_metric(self):
        config = ReconstructionConfig(threshold_metric=ThresholdMetric.RMSE)
        assert config.threshold_metric == ThresholdMetric.RMSE

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            ReconstructionConfig(threshold_metric="invalid")


class TestEvaluationConfig:
    def test_default_ratio(self):
        config = EvaluationConfig()
        assert config.threshold_ratio == 1.0

    def test_custom_ratio(self):
        config = EvaluationConfig(threshold_ratio=2.5)
        assert config.threshold_ratio == 2.5

    def test_zero_ratio_raises(self):
        with pytest.raises(ValueError):
            EvaluationConfig(threshold_ratio=0)

    def test_negative_ratio_raises(self):
        with pytest.raises(ValueError):
            EvaluationConfig(threshold_ratio=-1.0)


class TestComputeReconstructionErrors:
    def test_perfect_reconstruction(self):
        real = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        mse, mae, rmse, max_err = compute_reconstruction_errors(real, pred)

        assert all(v == 0.0 for v in mse)
        assert all(v == 0.0 for v in mae)
        assert all(v == 0.0 for v in rmse)
        assert all(v == 0.0 for v in max_err)

    def test_known_errors(self):
        real = np.array([[1.0, 0.0]])
        pred = np.array([[0.0, 0.0]])
        mse, mae, rmse, max_err = compute_reconstruction_errors(real, pred)

        assert mse[0] == pytest.approx(0.5)  # (1^2 + 0^2) / 2
        assert mae[0] == pytest.approx(0.5)  # (1 + 0) / 2
        assert rmse[0] == pytest.approx(np.sqrt(0.5))
        assert max_err[0] == pytest.approx(1.0)

    def test_output_lengths(self):
        real = np.random.randn(10, 5)
        pred = np.random.randn(10, 5)
        mse, mae, rmse, max_err = compute_reconstruction_errors(real, pred)

        assert len(mse) == 10
        assert len(mae) == 10
        assert len(rmse) == 10
        assert len(max_err) == 10


class TestCreateMetricsDataframe:
    def test_correct_columns(self):
        df = create_metrics_dataframe([0.1, 0.2], [0.05, 0.1], [0.31, 0.44], [0.2, 0.3])
        expected_cols = {
            "MSE_Recons_error",
            "MAE_Recons_error",
            "RMSE_Recons_error",
            "Max_Recons_error",
            "Score",
            "Score_Mean",
        }
        assert set(df.columns) == expected_cols

    def test_correct_length(self):
        df = create_metrics_dataframe([0.1], [0.05], [0.31], [0.2])
        assert len(df) == 1


class TestComputeColumnMetrics:
    def test_output_structure(self):
        real = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = np.array([[1.1, 2.2], [3.1, 4.2]])
        cols = ["feat_1", "feat_2"]
        result = compute_column_metrics(real, pred, cols, cluster_id=0)

        assert isinstance(result, pd.DataFrame)
        assert "Column" in result.columns
        assert "MSE" in result.columns
        assert "MAE" in result.columns
        assert result.iloc[-1]["Column"] == "Total Metrics"

    def test_perfect_reconstruction_columns(self):
        real = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = real.copy()
        result = compute_column_metrics(real, pred, ["a", "b"], 0)
        assert result["MSE"].iloc[0] == 0.0
        assert result["MAE"].iloc[0] == 0.0


class TestComputeThreshold:
    def test_basic(self):
        errors = pd.Series([1.0, 2.0, 3.0])
        # mean = 2.0, ratio = 1.0 => threshold = 2.0
        assert compute_threshold(errors, 1.0) == pytest.approx(2.0)

    def test_with_ratio(self):
        errors = pd.Series([1.0, 2.0, 3.0])
        # mean = 2.0, ratio = 1.5 => threshold = 3.0
        assert compute_threshold(errors, 1.5) == pytest.approx(3.0)

    def test_ratio_zero_gives_zero(self):
        errors = pd.Series([1.0, 2.0, 3.0])
        assert compute_threshold(errors, 0.0) == 0.0


class TestDetectAnomalies:
    def test_all_normal(self):
        errors = pd.Series([0.1, 0.2, 0.3])
        result = detect_anomalies(errors, threshold=1.0)
        assert (result == 0).all()

    def test_all_anomalies(self):
        errors = pd.Series([2.0, 3.0, 4.0])
        result = detect_anomalies(errors, threshold=1.0)
        assert (result == 1).all()

    def test_mixed(self):
        errors = pd.Series([0.5, 1.5, 0.3, 2.0])
        result = detect_anomalies(errors, threshold=1.0)
        expected = pd.Series([0, 1, 0, 1])
        pd.testing.assert_series_equal(result, expected)

    def test_exact_threshold_is_normal(self):
        errors = pd.Series([1.0])
        result = detect_anomalies(errors, threshold=1.0)
        assert result.iloc[0] == 0


class TestVectorizedErrorsMatchReference:
    """The vectorized implementation must equal the per-row scikit-learn definition."""

    def test_matches_sklearn_row_by_row(self):
        from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error

        rng = np.random.default_rng(0)
        real, pred = rng.normal(size=(50, 7)), rng.normal(size=(50, 7))
        mse, mae, rmse, max_err = compute_reconstruction_errors(real, pred)

        for i in range(len(real)):
            assert mse[i] == pytest.approx(mean_squared_error(real[i], pred[i]))
            assert mae[i] == pytest.approx(mean_absolute_error(real[i], pred[i]))
            assert rmse[i] == pytest.approx(np.sqrt(mean_squared_error(real[i], pred[i])))
            assert max_err[i] == pytest.approx(max_error(real[i], pred[i]))

    def test_returns_one_dimensional_arrays(self):
        results = compute_reconstruction_errors(np.zeros((4, 3)), np.ones((4, 3)))
        assert all(isinstance(r, np.ndarray) and r.shape == (4,) for r in results)

    def test_accepts_dataframes_and_float32(self):
        real = pd.DataFrame(np.ones((3, 2), dtype="float32"))
        pred = pd.DataFrame(np.zeros((3, 2), dtype="float32"))
        mse, mae, rmse, max_err = compute_reconstruction_errors(real, pred)
        assert mse.dtype == np.float64
        assert mse.tolist() == [1.0, 1.0, 1.0]

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_reconstruction_errors(np.zeros((3, 2)), np.zeros((3, 3)))


class TestMetricsDataframeScore:
    @pytest.mark.parametrize(
        "metric,column",
        [
            (ThresholdMetric.MSE, "MSE_Recons_error"),
            (ThresholdMetric.MAE, "MAE_Recons_error"),
            (ThresholdMetric.RMSE, "RMSE_Recons_error"),
            (ThresholdMetric.MAX_ERROR, "Max_Recons_error"),
        ],
    )
    def test_score_is_active_metric(self, metric, column):
        df = create_metrics_dataframe([1.0, 3.0], [2.0, 4.0], [5.0, 7.0], [6.0, 8.0], metric)
        assert df["Score"].tolist() == df[column].tolist()
        assert (df["Score_Mean"] == df[column].mean()).all()


class TestColumnMetricsValues:
    def test_known_values(self):
        real = np.array([[1.0, 0.0], [3.0, 0.0]])
        pred = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = compute_column_metrics(real, pred, ["a", "b"], cluster_id=2)

        first = result.iloc[0]
        assert first["MSE"] == pytest.approx(5.0)  # (1 + 9) / 2
        assert first["MAE"] == pytest.approx(2.0)
        assert first["RMSE"] == pytest.approx(np.sqrt(5.0))
        assert first["Max_Error"] == pytest.approx(3.0)
        assert first["partition"] == 2
        assert result.iloc[-1]["Max_Error"] == pytest.approx(3.0)


class TestDetectAnomaliesContainers:
    def test_numpy_input_returns_numpy(self):
        result = detect_anomalies(np.array([0.5, 1.5]), threshold=1.0)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [0, 1]

    def test_series_keeps_index(self):
        errors = pd.Series([0.5, 1.5], index=["x", "y"])
        result = detect_anomalies(errors, threshold=1.0)
        assert list(result.index) == ["x", "y"]
        assert result.dtype == np.int64

    def test_threshold_is_python_float(self):
        assert type(compute_threshold(pd.Series([1.0, 3.0]), 2)) is float


class TestEvaluationConfigNonFinite:
    @pytest.mark.parametrize("ratio", [float("nan"), float("inf")])
    def test_non_finite_ratio_raises(self, ratio):
        with pytest.raises(ValueError):
            EvaluationConfig(threshold_ratio=ratio)
