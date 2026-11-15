"""Tests for clustering module: models, registry, and orchestrator."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from segmentae.clusters.clustering import Clustering
from segmentae.clusters.models import (
    ClusteringConfig,
    GaussianMixtureCluster,
    KMeansCluster,
    MiniBatchKMeansCluster,
)
from segmentae.clusters.registry import ClusterRegistry
from segmentae.core.constants import ClusterModel
from segmentae.core.exceptions import ModelNotFittedError, ValidationError


class TestKMeansCluster:
    def test_fit_and_predict(self, sample_numerical_data):
        model = KMeansCluster(n_clusters=3)
        model.fit(sample_numerical_data)
        labels = model.predict(sample_numerical_data)

        assert model.is_fitted
        assert len(labels) == len(sample_numerical_data)
        assert set(labels).issubset({0, 1, 2})

    def test_fit_predict(self, sample_numerical_data):
        model = KMeansCluster(n_clusters=2)
        labels = model.fit_predict(sample_numerical_data)

        assert model.is_fitted
        assert len(labels) == len(sample_numerical_data)
        assert set(labels).issubset({0, 1})

    def test_n_clusters_property(self):
        model = KMeansCluster(n_clusters=5)
        assert model.n_clusters == 5

    def test_predict_before_fit_raises(self, sample_numerical_data):
        model = KMeansCluster()
        with pytest.raises(ModelNotFittedError):
            model.predict(sample_numerical_data)

    def test_invalid_n_clusters(self):
        with pytest.raises(PydanticValidationError):
            KMeansCluster(n_clusters=0)

    def test_invalid_input_type(self):
        model = KMeansCluster()
        with pytest.raises(ValidationError):
            model.fit([[1, 2], [3, 4]])


class TestMiniBatchKMeansCluster:
    def test_fit_and_predict(self, sample_numerical_data):
        model = MiniBatchKMeansCluster(n_clusters=3)
        model.fit(sample_numerical_data)
        labels = model.predict(sample_numerical_data)

        assert model.is_fitted
        assert len(labels) == len(sample_numerical_data)

    def test_fit_predict(self, sample_numerical_data):
        model = MiniBatchKMeansCluster(n_clusters=2)
        labels = model.fit_predict(sample_numerical_data)
        assert len(labels) == len(sample_numerical_data)


class TestGaussianMixtureCluster:
    def test_fit_and_predict(self, sample_numerical_data):
        model = GaussianMixtureCluster(n_components=3)
        model.fit(sample_numerical_data)
        labels = model.predict(sample_numerical_data)

        assert model.is_fitted
        assert len(labels) == len(sample_numerical_data)

    def test_n_clusters_returns_n_components(self):
        model = GaussianMixtureCluster(n_components=5)
        assert model.n_clusters == 5

    def test_invalid_covariance_type(self):
        with pytest.raises(PydanticValidationError):
            GaussianMixtureCluster(covariance_type="invalid")

    def test_valid_covariance_types(self, sample_numerical_data):
        for cov_type in ["full", "tied", "diag", "spherical"]:
            model = GaussianMixtureCluster(covariance_type=cov_type)
            model.fit(sample_numerical_data)
            assert model.is_fitted


class TestClusterRegistry:
    def test_registered_models(self):
        available = ClusterRegistry.list_available()
        assert ClusterModel.KMEANS in available
        assert ClusterModel.MINIBATCH_KMEANS in available
        assert ClusterModel.GMM in available

    def test_is_registered(self):
        assert ClusterRegistry.is_registered(ClusterModel.KMEANS)
        assert ClusterRegistry.is_registered(ClusterModel.GMM)

    def test_create_kmeans(self):
        model = ClusterRegistry.create(ClusterModel.KMEANS, n_clusters=4)
        assert isinstance(model, KMeansCluster)
        assert model.n_clusters == 4

    def test_create_gmm(self):
        model = ClusterRegistry.create(ClusterModel.GMM, n_components=2)
        assert isinstance(model, GaussianMixtureCluster)
        assert model.n_clusters == 2


class TestClusteringConfig:
    def test_valid_string_models(self):
        config = ClusteringConfig(cluster_models=["KMeans", "GMM"])
        assert len(config.cluster_models) == 2

    def test_valid_enum_models(self):
        config = ClusteringConfig(cluster_models=[ClusterModel.KMEANS])
        assert config.cluster_models[0] == ClusterModel.KMEANS

    def test_invalid_model_string(self):
        with pytest.raises(PydanticValidationError):
            ClusteringConfig(cluster_models=["InvalidModel"])

    def test_invalid_n_clusters(self):
        with pytest.raises(PydanticValidationError):
            ClusteringConfig(cluster_models=["KMeans"], n_clusters=0)

    def test_invalid_covariance_type(self):
        with pytest.raises(PydanticValidationError):
            ClusteringConfig(cluster_models=["GMM"], covariance_type="bad")


class TestClusteringOrchestrator:
    def test_fit_and_predict_kmeans(self, sample_numerical_data):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=3)
        cl.clustering_fit(sample_numerical_data)

        assert cl.is_fitted
        predictions = cl.cluster_prediction(sample_numerical_data)
        assert isinstance(predictions, pd.DataFrame)
        assert "KMeans" in predictions.columns
        assert len(predictions) == len(sample_numerical_data)

    def test_fit_and_predict_gmm(self, sample_numerical_data):
        cl = Clustering(cluster_model=["GMM"], n_clusters=2)
        cl.clustering_fit(sample_numerical_data)
        predictions = cl.cluster_prediction(sample_numerical_data)

        assert "GMM" in predictions.columns

    def test_fit_and_predict_minibatch(self, sample_numerical_data):
        cl = Clustering(cluster_model=["MiniBatchKMeans"], n_clusters=3)
        cl.clustering_fit(sample_numerical_data)
        predictions = cl.cluster_prediction(sample_numerical_data)

        assert "MiniBatchKMeans" in predictions.columns

    def test_predict_before_fit_raises(self, sample_numerical_data):
        cl = Clustering(cluster_model=["KMeans"])
        with pytest.raises(ModelNotFittedError):
            cl.cluster_prediction(sample_numerical_data)

    def test_invalid_input_type(self):
        cl = Clustering(cluster_model=["KMeans"])
        with pytest.raises(ValidationError):
            cl.clustering_fit(np.array([[1, 2], [3, 4]]))

    def test_empty_dataframe_raises(self):
        cl = Clustering(cluster_model=["KMeans"])
        with pytest.raises(ValidationError):
            cl.clustering_fit(pd.DataFrame())

    def test_fitted_models_property(self, sample_numerical_data):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=3)
        cl.clustering_fit(sample_numerical_data)
        models = cl.fitted_models
        assert "KMeans" in models

    def test_repr(self):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=3)
        repr_str = repr(cl)
        assert "Clustering" in repr_str
        assert "3" in repr_str

    def test_enum_based_init(self, sample_numerical_data):
        cl = Clustering(cluster_model=[ClusterModel.KMEANS], n_clusters=2)
        cl.clustering_fit(sample_numerical_data)
        assert cl.is_fitted


class TestClusterModelNames:
    def test_single_string_is_accepted(self, sample_numerical_data):
        cl = Clustering(cluster_model="KMeans", n_clusters=2)
        assert cl.cluster_model == ["KMeans"]
        cl.clustering_fit(sample_numerical_data)
        assert list(cl.cluster_prediction(sample_numerical_data).columns) == ["KMeans"]

    def test_single_enum_is_accepted(self):
        assert Clustering(cluster_model=ClusterModel.GMM).cluster_model == ["GMM"]

    def test_cluster_model_attribute_holds_plain_strings(self):
        cl = Clustering(cluster_model=[ClusterModel.KMEANS, "GMM"])
        assert cl.cluster_model == ["KMeans", "GMM"]
        assert all(type(name) is str for name in cl.cluster_model)
        assert "models=[KMeans, GMM]" in repr(cl)

    def test_default_model_is_kmeans(self):
        assert Clustering().cluster_model == ["KMeans"]

    def test_empty_model_list_raises(self):
        with pytest.raises(PydanticValidationError):
            Clustering(cluster_model=[])

    def test_unknown_model_lists_available_names(self):
        with pytest.raises(PydanticValidationError, match="KMeans"):
            Clustering(cluster_model=["DBSCAN"])


class TestClusteringBehaviour:
    def test_prediction_preserves_index(self, sample_numerical_data):
        data = sample_numerical_data.copy()
        data.index = data.index + 1000
        cl = Clustering(cluster_model=["KMeans", "GMM"], n_clusters=2).clustering_fit(data)
        predictions = cl.cluster_prediction(data)

        assert predictions.index.equals(data.index)
        assert list(predictions.columns) == ["KMeans", "GMM"]

    def test_clustering_fit_returns_self(self, sample_numerical_data):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=2)
        assert cl.clustering_fit(sample_numerical_data) is cl
        assert cl.feature_names == list(sample_numerical_data.columns)

    def test_same_random_state_gives_same_labels(self, sample_numerical_data):
        first = Clustering(cluster_model=["KMeans"], n_clusters=3, random_state=1)
        second = Clustering(cluster_model=["KMeans"], n_clusters=3, random_state=1)
        labels_a = first.clustering_fit(sample_numerical_data).cluster_prediction(
            sample_numerical_data
        )
        labels_b = second.clustering_fit(sample_numerical_data).cluster_prediction(
            sample_numerical_data
        )
        pd.testing.assert_frame_equal(labels_a, labels_b)

    def test_missing_values_raise(self, sample_data_with_nans):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=2)
        with pytest.raises(ValidationError, match="missing values"):
            cl.clustering_fit(sample_data_with_nans)

    def test_non_numeric_data_raises(self, sample_mixed_data):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=2)
        with pytest.raises(ValidationError, match="non-numeric"):
            cl.clustering_fit(sample_mixed_data)

    def test_prediction_with_missing_column_raises(self, sample_numerical_data):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=2).clustering_fit(
            sample_numerical_data
        )
        with pytest.raises(ValidationError, match="missing columns"):
            cl.cluster_prediction(sample_numerical_data.drop(columns=["feature_1"]))

    def test_prediction_realigns_reordered_columns(self, sample_numerical_data):
        cl = Clustering(cluster_model=["KMeans"], n_clusters=2).clustering_fit(
            sample_numerical_data
        )
        reference = cl.cluster_prediction(sample_numerical_data)
        reordered = cl.cluster_prediction(
            sample_numerical_data[sample_numerical_data.columns[::-1]]
        )
        pd.testing.assert_frame_equal(reordered, reference)

    def test_more_clusters_than_samples_raises(self, sample_numerical_data):
        from segmentae.core.exceptions import ClusteringError

        cl = Clustering(cluster_model=["KMeans"], n_clusters=10)
        with pytest.raises(ClusteringError, match="cannot form 10 clusters"):
            cl.clustering_fit(sample_numerical_data.head(5))
        assert not cl.is_fitted

    def test_backward_compatible_accessors(self, sample_numerical_data):
        cl = Clustering(cluster_model=["KMeans", "GMM"], n_clusters=2)
        assert cl.cmodel is None
        cl.clustering_fit(sample_numerical_data)
        assert set(cl.clustering_dict) == {"KMeans", "GMM"}
        assert isinstance(cl.cmodel, GaussianMixtureCluster)


class TestGaussianMixtureOptions:
    def test_init_params_is_passed_to_sklearn(self, sample_numerical_data):
        model = GaussianMixtureCluster(n_components=2, init_params="random")
        model.fit(sample_numerical_data)
        assert model._model.init_params == "random"

    def test_default_init_params_matches_sklearn_default(self):
        assert GaussianMixtureCluster().config.init_params == "kmeans"

    def test_invalid_init_params(self):
        with pytest.raises(PydanticValidationError):
            GaussianMixtureCluster(init_params="bogus")

    def test_orchestrator_forwards_covariance_type(self, sample_numerical_data):
        cl = Clustering(cluster_model=["GMM"], n_clusters=2, covariance_type="diag")
        cl.clustering_fit(sample_numerical_data)
        assert cl.fitted_models["GMM"]._model.covariance_type == "diag"
        assert cl.fitted_models["GMM"].n_clusters == 2


class TestRegistryExtensibility:
    """Custom algorithms can be registered by name and used end to end."""

    @pytest.fixture
    def birch_registered(self):
        from sklearn.cluster import Birch

        from segmentae.core.base import AbstractClusterModel

        class BirchCluster(AbstractClusterModel):
            def __init__(self, n_clusters=3, random_state=0):
                self._n_clusters = n_clusters
                self._model = None

            def fit(self, X):
                self._validate_input(X, "Training data")
                self._model = Birch(n_clusters=self._n_clusters).fit(X)

            def predict(self, X):
                self._validate_fitted()
                return self._model.predict(X)

            def fit_predict(self, X):
                self.fit(X)
                return self.predict(X)

            @property
            def n_clusters(self):
                return self._n_clusters

            @property
            def is_fitted(self):
                return self._model is not None

        ClusterRegistry.register("Birch", BirchCluster)
        yield BirchCluster
        ClusterRegistry.unregister("Birch")

    def test_custom_model_in_clustering(self, birch_registered, sample_numerical_data):
        assert ClusterRegistry.is_registered("Birch")
        cl = Clustering(cluster_model=["Birch"], n_clusters=2).clustering_fit(sample_numerical_data)
        predictions = cl.cluster_prediction(sample_numerical_data)
        assert list(predictions.columns) == ["Birch"]
        assert set(predictions["Birch"].unique()) == {0, 1}

    def test_custom_model_in_pipeline(self, birch_registered, two_cluster_data, stub_autoencoder):
        from segmentae.pipeline.segmentae import SegmentAE

        cl = Clustering(cluster_model=["Birch"], n_clusters=2).clustering_fit(two_cluster_data)
        pipeline = SegmentAE(ae_model=stub_autoencoder, cl_model=cl, verbose=False)
        pipeline.reconstruction(two_cluster_data)
        assert len(pipeline.detections(two_cluster_data)) == len(two_cluster_data)

    def test_unregister_removes_model(self, birch_registered):
        ClusterRegistry.unregister("Birch")
        assert not ClusterRegistry.is_registered("Birch")
        with pytest.raises(PydanticValidationError):
            Clustering(cluster_model=["Birch"])

    def test_duplicate_registration_raises_unless_overridden(self, birch_registered):
        from segmentae.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="already registered"):
            ClusterRegistry.register("Birch", birch_registered)
        ClusterRegistry.register("Birch", birch_registered, override=True)

    def test_non_cluster_class_is_rejected(self):
        from segmentae.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="AbstractClusterModel"):
            ClusterRegistry.register("Bogus", dict)

    def test_unknown_model_error_lists_options(self):
        from segmentae.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Valid options"):
            ClusterRegistry.create("Nope")

    def test_enum_and_string_keys_are_equivalent(self):
        assert ClusterRegistry.get("KMeans") is ClusterRegistry.get(ClusterModel.KMEANS)
        assert ClusterRegistry.list_available()[:3] == ["KMeans", "MiniBatchKMeans", "GMM"]


class TestReviewRegressions:
    def test_list_available_keeps_enum_members_for_builtins(self):
        available = ClusterRegistry.list_available()
        assert [model.value for model in available[:3]] == ["KMeans", "MiniBatchKMeans", "GMM"]
        assert all(isinstance(model, ClusterModel) for model in available[:3])
        assert ClusterRegistry.names()[:3] == ["KMeans", "MiniBatchKMeans", "GMM"]

    def test_duplicated_column_names_raise_validation_error(self, sample_numerical_data):
        duplicated = pd.concat(
            [sample_numerical_data, sample_numerical_data[["feature_1"]]], axis=1
        )
        with pytest.raises(ValidationError, match="duplicated column names"):
            Clustering(cluster_model=["KMeans"], n_clusters=2).clustering_fit(duplicated)

    def test_clustering_error_is_a_value_error(self):
        from segmentae.core.exceptions import ClusteringError

        assert issubclass(ClusteringError, ValueError)

    def test_model_returning_a_series_is_used_positionally(self, two_cluster_data):
        """Labels belong to rows by position, whatever container a custom model returns."""
        from segmentae.clusters.models import KMeansCluster

        class SeriesKMeans(KMeansCluster):
            def predict(self, X):
                return pd.Series(super().predict(X))  # fresh 0..n-1 index, like a careless model

        ClusterRegistry.register("SeriesKMeans", SeriesKMeans)
        try:
            shuffled = two_cluster_data.copy()
            shuffled.index = np.random.default_rng(0).permutation(len(shuffled))
            reference = Clustering(cluster_model=["KMeans"], n_clusters=2).clustering_fit(shuffled)
            custom = Clustering(cluster_model=["SeriesKMeans"], n_clusters=2).clustering_fit(
                shuffled
            )

            np.testing.assert_array_equal(
                custom.cluster_prediction(shuffled)["SeriesKMeans"].to_numpy(),
                reference.cluster_prediction(shuffled)["KMeans"].to_numpy(),
            )
        finally:
            ClusterRegistry.unregister("SeriesKMeans")
