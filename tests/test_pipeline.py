"""Tests for the SegmentAE pipeline (integration tests)."""

import numpy as np
import pandas as pd
import pytest

from segmentae.autoencoders.dense import DenseAutoencoder
from segmentae.clusters.clustering import Clustering
from segmentae.core.exceptions import (
    ConfigurationError,
    ModelNotFittedError,
    ValidationError,
)
from segmentae.pipeline.segmentae import SegmentAE


@pytest.fixture
def fitted_autoencoder(scaled_data):
    """Train a small DenseAutoencoder for pipeline tests."""
    ae = DenseAutoencoder(
        hidden_dims=[8, 4],
        epochs=10,
        val_size=0.2,
        stopping_patient=5,
    )
    ae.fit(scaled_data)
    return ae


@pytest.fixture
def fitted_clustering(scaled_data):
    """Fit a KMeans clustering model for pipeline tests."""
    cl = Clustering(cluster_model=["KMeans"], n_clusters=2)
    cl.clustering_fit(scaled_data)
    return cl


@pytest.fixture
def pipeline(fitted_autoencoder, fitted_clustering):
    """Create a SegmentAE pipeline."""
    return SegmentAE(ae_model=fitted_autoencoder, cl_model=fitted_clustering)


class TestSegmentAEInit:
    def test_valid_init(self, fitted_autoencoder, fitted_clustering):
        sg = SegmentAE(ae_model=fitted_autoencoder, cl_model=fitted_clustering)
        assert sg.ae_model is fitted_autoencoder
        assert sg.cl_model is fitted_clustering
        assert not sg._is_reconstruction_fitted

    def test_invalid_autoencoder_raises(self, fitted_clustering):
        with pytest.raises(ConfigurationError, match="predict"):
            SegmentAE(ae_model="not_a_model", cl_model=fitted_clustering)

    def test_invalid_clustering_type_raises(self, fitted_autoencoder):
        with pytest.raises(ConfigurationError):
            SegmentAE(ae_model=fitted_autoencoder, cl_model="not_clustering")

    def test_unfitted_clustering_raises(self, fitted_autoencoder):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=2)
        with pytest.raises(ModelNotFittedError):
            SegmentAE(ae_model=fitted_autoencoder, cl_model=cl)

    def test_repr(self, pipeline):
        repr_str = repr(pipeline)
        assert "SegmentAE" in repr_str
        assert "DenseAutoencoder" in repr_str


class TestReconstruction:
    def test_reconstruction_returns_self(self, pipeline, scaled_data):
        result = pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        assert result is pipeline
        assert pipeline._is_reconstruction_fitted

    def test_reconstruction_stores_results(self, pipeline, scaled_data):
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        assert len(pipeline.preds_train) > 0
        assert len(pipeline.reconstruction_eval) > 0

    def test_reconstruction_with_different_metrics(self, pipeline, scaled_data):
        for metric in ["mse", "mae", "rmse", "max_error"]:
            pipeline.reconstruction(input_data=scaled_data, threshold_metric=metric)
            assert pipeline._is_reconstruction_fitted

    def test_reconstruction_invalid_input_type(self, pipeline):
        with pytest.raises(ValidationError):
            pipeline.reconstruction(input_data=np.array([[1, 2]]))

    def test_reconstruction_empty_data(self, pipeline):
        with pytest.raises(ValidationError):
            pipeline.reconstruction(input_data=pd.DataFrame())

    def test_reconstruction_with_target(self, pipeline, scaled_data, binary_target):
        target = binary_target[: len(scaled_data)]
        pipeline.reconstruction(input_data=scaled_data, target_col=target, threshold_metric="mse")
        assert pipeline._is_reconstruction_fitted


class TestEvaluation:
    def test_evaluation_returns_dict(self, pipeline, scaled_data, binary_target):
        target = binary_target[: len(scaled_data)]
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        results = pipeline.evaluation(
            input_data=scaled_data, target_col=target, threshold_ratio=2.0
        )

        assert isinstance(results, dict)
        assert "global metrics" in results
        assert "clusters metrics" in results
        assert "confusion matrix" in results
        assert "y_true vs y_pred" in results

    def test_evaluation_global_metrics(self, pipeline, scaled_data, binary_target):
        target = binary_target[: len(scaled_data)]
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        results = pipeline.evaluation(
            input_data=scaled_data, target_col=target, threshold_ratio=2.0
        )

        global_metrics = results["global metrics"]
        assert "Accuracy" in global_metrics.columns
        assert "Precision" in global_metrics.columns
        assert "Recall" in global_metrics.columns
        assert "F1 Score" in global_metrics.columns
        assert "Threshold Ratio" in global_metrics.columns

    def test_evaluation_before_reconstruction_raises(self, pipeline, scaled_data, binary_target):
        target = binary_target[: len(scaled_data)]
        with pytest.raises(ModelNotFittedError):
            pipeline.evaluation(input_data=scaled_data, target_col=target, threshold_ratio=1.0)

    def test_evaluation_multiple_thresholds(self, pipeline, scaled_data, binary_target):
        target = binary_target[: len(scaled_data)]
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")

        results = []
        for ratio in [0.5, 1.0, 2.0, 3.0]:
            result = pipeline.evaluation(
                input_data=scaled_data, target_col=target, threshold_ratio=ratio
            )
            results.append(result["global metrics"])

        combined = pd.concat(results, ignore_index=True)
        assert len(combined) == 4

    def test_evaluation_invalid_target_type(self, pipeline, scaled_data):
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        with pytest.raises(ValidationError):
            pipeline.evaluation(input_data=scaled_data, target_col=[0, 1, 0], threshold_ratio=1.0)

    def test_evaluation_target_length_mismatch(self, pipeline, scaled_data):
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        short_target = pd.Series([0, 1, 0])
        with pytest.raises(ValidationError, match="length"):
            pipeline.evaluation(
                input_data=scaled_data, target_col=short_target, threshold_ratio=1.0
            )


class TestDetections:
    def test_detections_returns_dataframe(self, pipeline, scaled_data):
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        predictions = pipeline.detections(input_data=scaled_data, threshold_ratio=2.0)

        assert isinstance(predictions, pd.DataFrame)
        assert "Predicted Anomalies" in predictions.columns
        assert len(predictions) == len(scaled_data)

    def test_detections_binary_output(self, pipeline, scaled_data):
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        predictions = pipeline.detections(input_data=scaled_data, threshold_ratio=2.0)

        anomaly_values = predictions["Predicted Anomalies"].unique()
        assert set(anomaly_values).issubset({0, 1})

    def test_detections_before_reconstruction_raises(self, pipeline, scaled_data):
        with pytest.raises(ModelNotFittedError):
            pipeline.detections(input_data=scaled_data, threshold_ratio=1.0)

    def test_higher_threshold_fewer_anomalies(self, pipeline, scaled_data):
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")

        preds_low = pipeline.detections(input_data=scaled_data, threshold_ratio=0.5)
        preds_high = pipeline.detections(input_data=scaled_data, threshold_ratio=5.0)

        anomalies_low = preds_low["Predicted Anomalies"].sum()
        anomalies_high = preds_high["Predicted Anomalies"].sum()
        assert anomalies_high <= anomalies_low

    def test_detections_cluster_column(self, pipeline, scaled_data):
        pipeline.reconstruction(input_data=scaled_data, threshold_metric="mse")
        predictions = pipeline.detections(input_data=scaled_data, threshold_ratio=2.0)
        assert "Cluster" in predictions.columns
        assert set(predictions["Cluster"].unique()).issubset({0, 1})


class TestEndToEnd:
    """Full pipeline integration test."""

    def test_full_workflow(self, scaled_data, binary_target):
        """Test the complete reconstruction -> evaluation -> detection flow."""
        target = binary_target[: len(scaled_data)]

        # Split data
        train = scaled_data.iloc[:150].reset_index(drop=True)
        test = scaled_data.iloc[150:].reset_index(drop=True)
        y_test = target.iloc[150:].reset_index(drop=True)

        # Train autoencoder
        ae = DenseAutoencoder(hidden_dims=[8, 4], epochs=10, stopping_patient=5)
        ae.fit(train)

        # Fit clustering
        cl = Clustering(cluster_model=["KMeans"], n_clusters=2)
        cl.clustering_fit(train)

        # Create pipeline
        sg = SegmentAE(ae_model=ae, cl_model=cl)

        # Reconstruction
        sg.reconstruction(input_data=train, threshold_metric="mse")
        assert sg._is_reconstruction_fitted

        # Evaluation
        results = sg.evaluation(input_data=test, target_col=y_test, threshold_ratio=2.0)
        assert "global metrics" in results
        accuracy = results["global metrics"]["Accuracy"].iloc[0]
        assert 0 <= accuracy <= 1

        # Detection
        predictions = sg.detections(input_data=test, threshold_ratio=2.0)
        assert len(predictions) == len(test)
        assert "Predicted Anomalies" in predictions.columns


# --------------------------------------------------------------------------------------
# Behaviour tests on a deterministic autoencoder (reconstruction = 0.9 * x)
# --------------------------------------------------------------------------------------


def _fitted_stub_pipeline(data, stub, n_clusters=2, metric="mse"):
    clustering = Clustering(cluster_model=["KMeans"], n_clusters=n_clusters)
    clustering.clustering_fit(data)
    pipeline = SegmentAE(ae_model=stub, cl_model=clustering, verbose=False)
    pipeline.reconstruction(input_data=data, threshold_metric=metric)
    return pipeline


class TestReconstructionNumerics:
    def test_errors_match_closed_form(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        expected_mse = ((0.1 * two_cluster_data) ** 2).mean(axis=1)

        for cluster in pipeline.reconstruction_eval.values():
            np.testing.assert_allclose(
                cluster["metrics"]["MSE_Recons_error"].to_numpy(),
                expected_mse.iloc[cluster["positions"]].to_numpy(),
            )

    def test_baseline_threshold_is_cluster_mean_error(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        assert set(pipeline.thresholds) == {0, 1}
        for cluster_id, threshold in pipeline.thresholds.items():
            errors = pipeline.reconstruction_eval[cluster_id]["metrics"]["MSE_Recons_error"]
            assert threshold == pytest.approx(errors.mean())
        assert pipeline.get_thresholds(2.0)[0] == pytest.approx(2 * pipeline.thresholds[0])
        assert pipeline.threshold_metric == "mse"

    @pytest.mark.parametrize(
        "metric,column",
        [
            ("mse", "MSE_Recons_error"),
            ("mae", "MAE_Recons_error"),
            ("rmse", "RMSE_Recons_error"),
            ("max_error", "Max_Recons_error"),
        ],
    )
    def test_score_column_follows_threshold_metric(
        self, two_cluster_data, stub_autoencoder, metric, column
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder, metric=metric)
        for cluster in pipeline.reconstruction_eval.values():
            pd.testing.assert_series_equal(
                cluster["metrics"]["Score"], cluster["metrics"][column], check_names=False
            )

    def test_autoencoder_is_called_once_per_dataset(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder, n_clusters=3)
        assert stub_autoencoder.calls == 1  # not once per cluster
        pipeline.detections(two_cluster_data)
        assert stub_autoencoder.calls == 2

    def test_clusters_partition_the_input(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder, n_clusters=3)
        positions = np.concatenate([c["positions"] for c in pipeline.preds_train.values()])
        assert sorted(positions.tolist()) == list(range(len(two_cluster_data)))


class TestIndexHandling:
    """Rows are handled by position: the index of the input never changes the result."""

    def test_relabelled_index_gives_identical_predictions(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        reference = pipeline.detections(two_cluster_data, threshold_ratio=1.0)

        relabelled = two_cluster_data.copy()
        relabelled.index = np.random.default_rng(1).permutation(len(relabelled))
        result = pipeline.detections(relabelled, threshold_ratio=1.0)

        assert result.index.equals(relabelled.index)
        np.testing.assert_array_equal(
            result["Predicted Anomalies"].to_numpy(), reference["Predicted Anomalies"].to_numpy()
        )
        np.testing.assert_array_equal(result["Cluster"].to_numpy(), reference["Cluster"].to_numpy())

    def test_split_without_index_reset_works_end_to_end(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        from sklearn.model_selection import train_test_split

        X_train, X_test, _, y_test = train_test_split(
            two_cluster_data, two_cluster_labels, train_size=0.7, random_state=0
        )
        pipeline = _fitted_stub_pipeline(X_train, stub_autoencoder)

        results = pipeline.evaluation(X_test, y_test, threshold_ratio=1.0)
        assert results["y_true vs y_pred"].index.equals(X_test.index)
        np.testing.assert_array_equal(
            results["y_true vs y_pred"]["y_test"].to_numpy(), y_test.to_numpy()
        )

        predictions = pipeline.detections(X_test, threshold_ratio=1.0)
        assert predictions.index.equals(X_test.index)
        np.testing.assert_array_equal(
            predictions["Predicted Anomalies"].to_numpy(),
            results["y_true vs y_pred"]["Predicted Anomalies"].to_numpy(),
        )

    def test_evaluation_metrics_independent_of_index(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        reference = pipeline.evaluation(two_cluster_data, two_cluster_labels)["global metrics"]

        data, labels = two_cluster_data.copy(), two_cluster_labels.copy()
        data.index = labels.index = np.random.default_rng(2).permutation(len(data))
        relabelled = pipeline.evaluation(data, labels)["global metrics"]

        pd.testing.assert_frame_equal(relabelled, reference)

    def test_string_index_is_supported(self, two_cluster_data, stub_autoencoder):
        data = two_cluster_data.copy()
        data.index = [f"row-{i}" for i in range(len(data))]
        pipeline = _fitted_stub_pipeline(data, stub_autoencoder)
        predictions = pipeline.detections(data)
        assert list(predictions.index) == list(data.index)


class TestDetectionOutput:
    def test_detection_columns(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        predictions = pipeline.detections(two_cluster_data, threshold_ratio=1.5)

        assert list(predictions.columns) == list(two_cluster_data.columns) + [
            "Predicted Anomalies",
            "Cluster",
            "Reconstruction Error",
            "Threshold",
        ]
        # Reconstructed feature values, in input order
        np.testing.assert_allclose(
            predictions[list("abcd")].to_numpy(), 0.9 * two_cluster_data.to_numpy()
        )
        # The flag is exactly "score above the row's threshold"
        expected = (predictions["Reconstruction Error"] > predictions["Threshold"]).astype(int)
        np.testing.assert_array_equal(predictions["Predicted Anomalies"], expected)

    def test_threshold_column_uses_cluster_threshold(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        predictions = pipeline.detections(two_cluster_data, threshold_ratio=2.0)
        thresholds = pipeline.get_thresholds(2.0)
        for cluster_id, rows in predictions.groupby("Cluster"):
            assert rows["Threshold"].nunique() == 1
            assert rows["Threshold"].iloc[0] == pytest.approx(thresholds[cluster_id])

    @pytest.mark.parametrize("ratio", [0, -1.0, float("nan"), float("inf")])
    def test_invalid_threshold_ratio_raises(self, two_cluster_data, stub_autoencoder, ratio):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        with pytest.raises(ValueError):
            pipeline.detections(two_cluster_data, threshold_ratio=ratio)
        with pytest.raises(ValueError):
            pipeline.evaluation(
                two_cluster_data, pd.Series(np.zeros(len(two_cluster_data), dtype=int)), ratio
            )


class TestClustersWithoutTrainingReference:
    """A cluster that received no training sample must not cause rows to be dropped."""

    @pytest.fixture
    def partial_pipeline(self, two_cluster_data, stub_autoencoder):
        clustering = Clustering(cluster_model=["KMeans"], n_clusters=2)
        clustering.clustering_fit(two_cluster_data)
        labels = clustering.cluster_prediction(two_cluster_data).iloc[:, 0]
        one_cluster_only = two_cluster_data[labels == labels.iloc[0]]

        pipeline = SegmentAE(ae_model=stub_autoencoder, cl_model=clustering, verbose=False)
        pipeline.reconstruction(input_data=one_cluster_only)
        assert len(pipeline.thresholds) == 1
        return pipeline

    def test_detections_score_every_row(self, partial_pipeline, two_cluster_data):
        with pytest.warns(UserWarning, match="no training reference"):
            predictions = partial_pipeline.detections(two_cluster_data)

        assert len(predictions) == len(two_cluster_data)
        unseen = predictions[~predictions["Cluster"].isin(partial_pipeline.thresholds)]
        assert len(unseen) > 0
        np.testing.assert_allclose(unseen["Threshold"], partial_pipeline.global_threshold)

    def test_evaluation_scores_every_row(self, partial_pipeline, two_cluster_data):
        labels = pd.Series(np.zeros(len(two_cluster_data), dtype=int))
        with pytest.warns(UserWarning, match="no training reference"):
            results = partial_pipeline.evaluation(two_cluster_data, labels)

        assert len(results["y_true vs y_pred"]) == len(two_cluster_data)
        assert len(results["clusters metrics"]) == 2


class TestEvaluationRobustness:
    def test_confusion_matrix_always_2x2(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        single_class = pd.Series(np.zeros(len(two_cluster_data), dtype=int))
        results = pipeline.evaluation(two_cluster_data, single_class, threshold_ratio=1e6)

        for cluster_id, entry in results["confusion matrix"].items():
            assert entry[f"CM_{cluster_id}"].shape == (2, 2)

    def test_confusion_matrices_sum_to_sample_count(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        results = pipeline.evaluation(two_cluster_data, two_cluster_labels)
        total = sum(entry[f"CM_{cid}"].sum() for cid, entry in results["confusion matrix"].items())
        assert total == len(two_cluster_data)

    def test_non_binary_target_raises(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        labels = pd.Series(np.arange(len(two_cluster_data)) % 3)
        with pytest.raises(ValidationError, match="binary"):
            pipeline.evaluation(two_cluster_data, labels)

    def test_boolean_target_is_accepted(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        as_int = pipeline.evaluation(two_cluster_data, two_cluster_labels)["global metrics"]
        as_bool = pipeline.evaluation(two_cluster_data, two_cluster_labels.astype(bool))[
            "global metrics"
        ]
        pd.testing.assert_frame_equal(as_int, as_bool)

    def test_threshold_sweep_reuses_reconstruction(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        calls = stub_autoencoder.calls
        for ratio in [0.5, 1.0, 2.0]:
            results = pipeline.evaluation(two_cluster_data, two_cluster_labels, ratio)
            assert results["global metrics"]["Threshold Ratio"].iloc[0] == ratio
        assert stub_autoencoder.calls == calls + 1

    def test_higher_ratio_never_flags_more(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        flagged = [
            pipeline.evaluation(two_cluster_data, two_cluster_labels, ratio)["y_true vs y_pred"][
                "Predicted Anomalies"
            ].sum()
            for ratio in [0.5, 1.0, 1.5, 3.0]
        ]
        assert flagged == sorted(flagged, reverse=True)

    def test_new_reference_invalidates_previous_results(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        pipeline.evaluation(two_cluster_data, two_cluster_labels)
        pipeline.reconstruction(two_cluster_data, threshold_metric="mae")
        assert pipeline.results == {}
        assert pipeline.reconstruction_test == {}

        results = pipeline.evaluation(two_cluster_data, two_cluster_labels)
        assert results["global metrics"]["Threshold Metric"].iloc[0] == "MAE"

    def test_failed_evaluation_leaves_pipeline_usable(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        broken = two_cluster_data.copy()
        broken.iloc[0, 0] = np.nan
        with pytest.raises(ValidationError, match="missing values"):
            pipeline.evaluation(broken, pd.Series(np.zeros(len(broken), dtype=int)))

        assert pipeline.reconstruction(two_cluster_data) is pipeline
        assert len(pipeline.detections(two_cluster_data)) == len(two_cluster_data)


class TestInputValidation:
    def test_missing_feature_column_raises(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        with pytest.raises(ValidationError, match="missing columns"):
            pipeline.detections(two_cluster_data.drop(columns=["a"]))

    def test_reordered_columns_are_realigned(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        reference = pipeline.detections(two_cluster_data)
        reordered = pipeline.detections(two_cluster_data[list("dcba")])
        pd.testing.assert_frame_equal(reordered, reference)

    def test_non_numeric_input_raises(self, two_cluster_data, stub_autoencoder):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        text = two_cluster_data.astype({"a": str})
        with pytest.raises(ValidationError, match="non-numeric"):
            pipeline.detections(text)

    @pytest.mark.parametrize("name", ["Cluster", "Threshold", "Predicted Anomalies"])
    def test_feature_named_like_an_output_column_raises(
        self, two_cluster_data, stub_autoencoder, name
    ):
        data = two_cluster_data.rename(columns={"a": name})
        clustering = Clustering(cluster_model=["KMeans"], n_clusters=2)
        clustering.clustering_fit(data)
        pipeline = SegmentAE(ae_model=stub_autoencoder, cl_model=clustering, verbose=False)
        with pytest.raises(ValidationError, match="clash with the columns added"):
            pipeline.reconstruction(data)

    def test_non_finite_reconstruction_raises(self, two_cluster_data):
        from segmentae.core.exceptions import AutoencoderError

        class DivergedAutoencoder:
            def predict(self, X):
                return np.full(np.asarray(X).shape, np.nan)

        clustering = Clustering(cluster_model=["KMeans"], n_clusters=2)
        clustering.clustering_fit(two_cluster_data)
        pipeline = SegmentAE(ae_model=DivergedAutoencoder(), cl_model=clustering)
        with pytest.raises(AutoencoderError, match="non-finite"):
            pipeline.reconstruction(two_cluster_data)

    def test_wrong_reconstruction_shape_raises(self, two_cluster_data):
        from segmentae.core.exceptions import AutoencoderError

        class WrongShapeAutoencoder:
            def predict(self, X):
                return np.asarray(X)[:, :2]

        clustering = Clustering(cluster_model=["KMeans"], n_clusters=2)
        clustering.clustering_fit(two_cluster_data)
        pipeline = SegmentAE(ae_model=WrongShapeAutoencoder(), cl_model=clustering)
        with pytest.raises(AutoencoderError, match="shape"):
            pipeline.reconstruction(two_cluster_data)

    def test_predict_attribute_must_be_callable(self, two_cluster_data):
        class NotCallable:
            predict = "nope"

        clustering = Clustering(cluster_model=["KMeans"], n_clusters=2)
        clustering.clustering_fit(two_cluster_data)
        with pytest.raises(ConfigurationError, match="predict"):
            SegmentAE(ae_model=NotCallable(), cl_model=clustering)

    def test_multiple_cluster_models_warn(self, two_cluster_data, stub_autoencoder):
        clustering = Clustering(cluster_model=["KMeans", "GMM"], n_clusters=2)
        clustering.clustering_fit(two_cluster_data)
        with pytest.warns(UserWarning, match="only uses the first"):
            SegmentAE(ae_model=stub_autoencoder, cl_model=clustering)

    def test_verbose_controls_threshold_printing(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder, capsys
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        pipeline.evaluation(two_cluster_data, two_cluster_labels)
        assert capsys.readouterr().out == ""

        pipeline.verbose = True
        pipeline.evaluation(two_cluster_data, two_cluster_labels)
        assert "Reconstruction Threshold" in capsys.readouterr().out


class TestTargetAlignment:
    """Labels follow their rows by index label when possible, otherwise by position."""

    def test_reordered_target_with_same_labels_is_aligned_by_label(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        reference = pipeline.evaluation(two_cluster_data, two_cluster_labels)

        reordered = two_cluster_labels.sample(frac=1, random_state=0)  # same labels, other order
        results = pipeline.evaluation(two_cluster_data, reordered)

        pd.testing.assert_frame_equal(results["global metrics"], reference["global metrics"])
        pd.testing.assert_frame_equal(results["y_true vs y_pred"], reference["y_true vs y_pred"])

    def test_target_with_unrelated_index_is_paired_by_position(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        reference = pipeline.evaluation(two_cluster_data, two_cluster_labels)["global metrics"]

        shifted = two_cluster_labels.copy()
        shifted.index = shifted.index + 10_000
        results = pipeline.evaluation(two_cluster_data, shifted)["global metrics"]
        pd.testing.assert_frame_equal(results, reference)

    def test_training_labels_are_stored_with_their_rows(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        clustering = Clustering(cluster_model=["KMeans"], n_clusters=2).clustering_fit(
            two_cluster_data
        )
        pipeline = SegmentAE(ae_model=stub_autoencoder, cl_model=clustering, verbose=False)
        reordered = two_cluster_labels.sample(frac=1, random_state=1)
        pipeline.reconstruction(two_cluster_data, target_col=reordered)

        for cluster in pipeline.preds_train.values():
            np.testing.assert_array_equal(
                cluster["y_true"].to_numpy(),
                two_cluster_labels.iloc[cluster["positions"]].to_numpy(),
            )


class TestIndexAndShapeEdgeCases:
    def test_multiindex_frame_is_supported(
        self, two_cluster_data, two_cluster_labels, stub_autoencoder
    ):
        data = two_cluster_data.copy()
        data.index = pd.MultiIndex.from_arrays(
            [np.arange(len(data)) // 10, np.arange(len(data)) % 10]
        )
        pipeline = _fitted_stub_pipeline(data, stub_autoencoder)

        results = pipeline.evaluation(data, two_cluster_labels)
        assert results["y_true vs y_pred"].index.equals(data.index)
        assert pipeline.detections(data).index.equals(data.index)

    def test_single_feature_model_may_return_a_flat_array(self):
        class FlatAutoencoder:
            def predict(self, X):
                return np.asarray(X, dtype=float).ravel() * 0.9

        data = pd.DataFrame({"only": np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])})
        clustering = Clustering(cluster_model=["KMeans"], n_clusters=2).clustering_fit(data)
        pipeline = SegmentAE(ae_model=FlatAutoencoder(), cl_model=clustering, verbose=False)
        pipeline.reconstruction(data)

        predictions = pipeline.detections(data)
        np.testing.assert_allclose(predictions["only"].to_numpy(), 0.9 * data["only"].to_numpy())

    @pytest.mark.parametrize("ratio", ["2", True, None])
    def test_threshold_ratio_must_be_a_number(self, two_cluster_data, stub_autoencoder, ratio):
        pipeline = _fitted_stub_pipeline(two_cluster_data, stub_autoencoder)
        with pytest.raises(ValidationError, match="threshold_ratio"):
            pipeline.detections(two_cluster_data, threshold_ratio=ratio)
        with pytest.raises(ValidationError, match="threshold_ratio"):
            pipeline.get_thresholds(ratio)
