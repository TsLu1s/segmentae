"""Tests for the grid search optimizer."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from segmentae.core.constants import ClusterModel, PerformanceMetric
from segmentae.core.exceptions import ConfigurationError, ValidationError
from segmentae.optimization import OptimizerConfig, SegmentAE_Optimizer
from segmentae.pipeline.segmentae import SegmentAE

from .conftest import StubAutoencoder


@pytest.fixture
def labeled_test_data(two_cluster_data):
    """Test set with 30 injected anomalies (uniform noise) and their labels."""
    rng = np.random.default_rng(1)
    data = two_cluster_data.sample(n=200, random_state=1).reset_index(drop=True)
    labels = pd.Series(np.zeros(len(data), dtype=int))
    anomalies = rng.choice(len(data), 30, replace=False)
    data.iloc[anomalies] = rng.uniform(0, 1, size=(30, data.shape[1]))
    labels.iloc[anomalies] = 1
    return data, labels


def _optimizer(**overrides):
    params = dict(
        autoencoder_models=[StubAutoencoder()],
        n_clusters_list=[1, 2],
        cluster_models=["KMeans"],
        threshold_ratios=[1, 2],
        performance_metric="f1_score",
        verbose=False,
    )
    params.update(overrides)
    return SegmentAE_Optimizer(**params)


class TestOptimizerConfig:
    def test_defaults(self):
        config = OptimizerConfig()
        assert config.n_clusters_list == [1, 2, 3, 4]
        assert config.cluster_models == ["KMeans", "MiniBatchKMeans", "GMM"]
        assert config.performance_metric == PerformanceMetric.F1_SCORE
        assert config.threshold_metric == "mse"

    def test_enum_cluster_models_become_names(self):
        config = OptimizerConfig(cluster_models=[ClusterModel.KMEANS, ClusterModel.GMM])
        assert config.cluster_models == ["KMeans", "GMM"]
        assert all(type(name) is str for name in config.cluster_models)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("f1_score", PerformanceMetric.F1_SCORE),
            ("F1 Score", PerformanceMetric.F1_SCORE),
            ("f1", PerformanceMetric.F1_SCORE),
            ("Accuracy", PerformanceMetric.ACCURACY),
            ("accuracy", PerformanceMetric.ACCURACY),
            ("PRECISION", PerformanceMetric.PRECISION),
            ("recall", PerformanceMetric.RECALL),
            (PerformanceMetric.RECALL, PerformanceMetric.RECALL),
        ],
    )
    def test_performance_metric_spellings(self, name, expected):
        assert OptimizerConfig(performance_metric=name).performance_metric == expected

    @pytest.mark.parametrize(
        "params",
        [
            {"cluster_models": ["NotAModel"]},
            {"cluster_models": []},
            {"n_clusters_list": [0]},
            {"n_clusters_list": []},
            {"threshold_ratios": [1, -1]},
            {"threshold_ratios": []},
            {"performance_metric": "auc"},
            {"threshold_metric": "huber"},
        ],
    )
    def test_invalid_configuration_raises(self, params):
        with pytest.raises(PydanticValidationError):
            OptimizerConfig(**params)


class TestOptimizerInit:
    def test_empty_autoencoders_raises(self):
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            SegmentAE_Optimizer(autoencoder_models=[])

    def test_autoencoder_without_predict_raises(self):
        with pytest.raises(ConfigurationError, match="predict"):
            SegmentAE_Optimizer(autoencoder_models=[object()])

    def test_single_autoencoder_is_wrapped_in_list(self):
        stub = StubAutoencoder()
        optimizer = SegmentAE_Optimizer(autoencoder_models=stub)
        assert optimizer.autoencoder_models == [stub]

    def test_configuration_is_exposed(self):
        optimizer = _optimizer(cluster_models=[ClusterModel.GMM], threshold_metric="mae")
        assert optimizer.n_clusters_list == [1, 2]
        assert optimizer.cluster_models == ["GMM"]
        assert optimizer.threshold_ratios == [1, 2]
        assert optimizer.threshold_metric == "mae"
        assert optimizer.best_model is None
        assert optimizer.best_params is None

    def test_repr(self):
        assert "SegmentAE_Optimizer" in repr(_optimizer())


class TestOptimize:
    def test_returns_best_pipeline_and_leaderboard(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        optimizer = _optimizer(cluster_models=["KMeans", "GMM"])
        best = optimizer.optimize(two_cluster_data, X_test, y_test)

        assert isinstance(best, SegmentAE)
        assert best is optimizer.optimal_segmentae
        # 2 n_clusters x 2 cluster models x 1 autoencoder x 2 threshold ratios
        assert len(optimizer.leaderboard) == 8
        for column in ["Autoencoder", "Cluster", "N_Clusters", "Threshold Ratio", "F1 Score"]:
            assert column in optimizer.leaderboard.columns

    def test_leaderboard_sorted_and_consistent_with_best(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        optimizer = _optimizer(threshold_ratios=[0.5, 1, 2, 4])
        optimizer.optimize(two_cluster_data, X_test, y_test)

        scores = optimizer.leaderboard["F1 Score"]
        assert scores.is_monotonic_decreasing
        assert optimizer.best_performance == pytest.approx(scores.iloc[0])

        top = optimizer.leaderboard.iloc[0]
        assert top["Threshold Ratio"] == optimizer.best_threshold_ratio
        assert top["N_Clusters"] == optimizer.best_n_clusters
        assert top["Cluster"] == optimizer.best_cluster_model

    def test_returned_pipeline_results_match_best_ratio(self, two_cluster_data, labeled_test_data):
        """The best pipeline must not be left in the state of the last ratio tried."""
        X_test, y_test = labeled_test_data
        optimizer = _optimizer(n_clusters_list=[2], threshold_ratios=[0.5, 1, 2, 50])
        best = optimizer.optimize(two_cluster_data, X_test, y_test)

        assert optimizer.best_threshold_ratio != 50
        metrics = best.results["global metrics"]
        assert metrics["Threshold Ratio"].iloc[0] == optimizer.best_threshold_ratio
        assert metrics["F1 Score"].iloc[0] == pytest.approx(optimizer.best_performance)

    def test_second_run_starts_from_clean_state(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        optimizer = _optimizer(performance_metric="accuracy")
        optimizer.optimize(two_cluster_data, X_test, y_test)
        first_best = optimizer.best_performance

        # Flipping the labels makes every configuration score far lower
        optimizer.optimize(two_cluster_data, X_test, 1 - y_test)
        assert optimizer.best_performance < first_best
        assert optimizer.best_performance == pytest.approx(
            optimizer.leaderboard["Accuracy"].iloc[0]
        )

    def test_each_autoencoder_reconstructs_each_dataset_once(
        self, two_cluster_data, labeled_test_data
    ):
        X_test, y_test = labeled_test_data
        stub = StubAutoencoder()
        optimizer = _optimizer(
            autoencoder_models=[stub],
            n_clusters_list=[1, 2, 3],
            cluster_models=["KMeans", "MiniBatchKMeans", "GMM"],
            threshold_ratios=[1, 2, 3],
        )
        optimizer.optimize(two_cluster_data, X_test, y_test)

        assert len(optimizer.leaderboard) == 27
        assert stub.calls == 2  # once for X_train, once for X_test
        assert optimizer.optimal_segmentae._reconstruction_cache is None

    def test_scores_match_a_manually_built_pipeline(self, two_cluster_data, labeled_test_data):
        """Sharing reconstructions across configurations must not change any result."""
        from segmentae.clusters.clustering import Clustering

        X_test, y_test = labeled_test_data
        optimizer = _optimizer(n_clusters_list=[2], cluster_models=["GMM"], threshold_ratios=[1.5])
        optimizer.optimize(two_cluster_data, X_test, y_test)

        clustering = Clustering(cluster_model=["GMM"], n_clusters=2).clustering_fit(
            two_cluster_data
        )
        manual = SegmentAE(StubAutoencoder(), clustering, verbose=False)
        manual.reconstruction(two_cluster_data)
        expected = manual.evaluation(X_test, y_test, threshold_ratio=1.5)["global metrics"]

        row = optimizer.leaderboard.iloc[[0]].reset_index(drop=True)
        pd.testing.assert_frame_equal(row[expected.columns], expected)

    def test_autoencoders_of_same_class_are_distinguishable(
        self, two_cluster_data, labeled_test_data
    ):
        X_test, y_test = labeled_test_data
        good, bad = StubAutoencoder(0.9), StubAutoencoder(0.0)
        optimizer = _optimizer(autoencoder_models=[bad, good], n_clusters_list=[2])
        optimizer.optimize(two_cluster_data, X_test, y_test)

        assert set(optimizer.leaderboard["Autoencoder_Index"]) == {0, 1}
        assert optimizer.best_model == "StubAutoencoder"
        assert optimizer.best_autoencoder in (good, bad)
        assert optimizer.best_params["autoencoder_index"] == optimizer.autoencoder_models.index(
            optimizer.best_autoencoder
        )

    def test_best_params(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        optimizer = _optimizer(threshold_metric="mae")
        optimizer.optimize(two_cluster_data, X_test, y_test)
        params = optimizer.best_params

        assert params["cluster_model"] == "KMeans"
        assert params["n_clusters"] in (1, 2)
        assert params["threshold_ratio"] in (1, 2)
        assert params["threshold_metric"] == "mae"
        assert set(optimizer.leaderboard["Threshold Metric"]) == {"MAE"}

    def test_enum_cluster_models_give_string_results(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        optimizer = _optimizer(cluster_models=[ClusterModel.KMEANS, ClusterModel.GMM])
        optimizer.optimize(two_cluster_data, X_test, y_test)

        assert set(optimizer.leaderboard["Cluster"]) == {"KMeans", "GMM"}
        assert type(optimizer.best_cluster_model) is str

    def test_verbose_false_prints_nothing(self, two_cluster_data, labeled_test_data, capsys):
        X_test, y_test = labeled_test_data
        _optimizer().optimize(two_cluster_data, X_test, y_test)
        assert capsys.readouterr().out == ""

    def test_verbose_true_prints_progress(self, two_cluster_data, labeled_test_data, capsys):
        X_test, y_test = labeled_test_data
        _optimizer(verbose=True).optimize(two_cluster_data, X_test, y_test)
        output = capsys.readouterr().out
        assert "Starting Grid Search Optimization" in output
        assert "OPTIMIZATION COMPLETE" in output
        assert "Cluster Model: KMeans" in output

    def test_detections_with_best_pipeline(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        optimizer = _optimizer()
        best = optimizer.optimize(two_cluster_data, X_test, y_test)
        predictions = best.detections(X_test, threshold_ratio=optimizer.best_threshold_ratio)

        np.testing.assert_array_equal(
            predictions["Predicted Anomalies"].to_numpy(),
            best.results["y_true vs y_pred"]["Predicted Anomalies"].to_numpy(),
        )


class TestOptimizeInputValidation:
    def test_non_dataframe_train_raises(self, labeled_test_data):
        X_test, y_test = labeled_test_data
        with pytest.raises(ValidationError, match="X_train"):
            _optimizer().optimize(X_test.to_numpy(), X_test, y_test)

    def test_target_must_be_series(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        with pytest.raises(ValidationError, match="y_test"):
            _optimizer().optimize(two_cluster_data, X_test, y_test.to_numpy())

    def test_target_length_mismatch_raises(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        with pytest.raises(ValidationError, match="length"):
            _optimizer().optimize(two_cluster_data, X_test, y_test.iloc[:10])

    def test_column_mismatch_raises(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        with pytest.raises(ValidationError, match="X_test"):
            _optimizer().optimize(two_cluster_data, X_test.drop(columns=["a"]), y_test)

    def test_non_binary_target_raises(self, two_cluster_data, labeled_test_data):
        X_test, y_test = labeled_test_data
        with pytest.raises(ValidationError, match="binary"):
            _optimizer().optimize(two_cluster_data, X_test, y_test + 1)


class TestOptimizerRobustness:
    def test_failed_search_leaves_no_partial_result(self, two_cluster_data, labeled_test_data):
        """If a model fails mid-search, the optimizer must not look as if it had finished."""
        X_test, y_test = labeled_test_data

        class FailsOnSecondDataset(StubAutoencoder):
            def predict(self, X, verbose=0):
                if self.calls >= 1:
                    raise RuntimeError("inference failed")
                return super().predict(X)

        optimizer = _optimizer(autoencoder_models=[StubAutoencoder(), FailsOnSecondDataset()])
        with pytest.raises(Exception, match="inference failed"):
            optimizer.optimize(two_cluster_data, X_test, y_test)

        assert optimizer.optimal_segmentae is None
        assert optimizer.leaderboard is None
        assert optimizer.best_threshold_ratio is None
        assert optimizer.best_performance == float("-inf")

    def test_reordered_test_columns_do_not_repeat_inference(
        self, two_cluster_data, labeled_test_data
    ):
        X_test, y_test = labeled_test_data
        reference = _optimizer()
        reference.optimize(two_cluster_data, X_test, y_test)

        stub = StubAutoencoder()
        optimizer = _optimizer(autoencoder_models=[stub])
        optimizer.optimize(two_cluster_data, X_test[list("dcba")], y_test)

        assert stub.calls == 2  # one per dataset, as with matching column order
        pd.testing.assert_frame_equal(optimizer.leaderboard, reference.leaderboard)
