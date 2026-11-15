"""Tests for core module: constants, exceptions, and enums."""

import pytest

from segmentae.core.constants import (
    ENCODER_CLASS_MAP,
    IMPUTER_CLASS_MAP,
    METRIC_COLUMN_MAP,
    SCALER_CLASS_MAP,
    ClusterModel,
    EncoderType,
    ImputerType,
    PhaseType,
    ScalerType,
    ThresholdMetric,
    get_metric_column_name,
    parse_threshold_metric,
)
from segmentae.core.exceptions import (
    AutoencoderError,
    ClusteringError,
    ConfigurationError,
    ModelNotFittedError,
    ReconstructionError,
    SegmentAEError,
    ValidationError,
)


class TestEnums:
    def test_phase_type_values(self):
        assert PhaseType.EVALUATION == "evaluation"
        assert PhaseType.TESTING == "testing"
        assert PhaseType.PREDICTION == "prediction"

    def test_cluster_model_values(self):
        assert ClusterModel.KMEANS == "KMeans"
        assert ClusterModel.MINIBATCH_KMEANS == "MiniBatchKMeans"
        assert ClusterModel.GMM == "GMM"
        assert len(ClusterModel) == 3

    def test_threshold_metric_values(self):
        assert ThresholdMetric.MSE == "mse"
        assert ThresholdMetric.MAE == "mae"
        assert ThresholdMetric.RMSE == "rmse"
        assert ThresholdMetric.MAX_ERROR == "max_error"

    def test_encoder_type_values(self):
        assert EncoderType.IFREQUENCY == "IFrequencyEncoder"
        assert EncoderType.LABEL == "LabelEncoder"
        assert EncoderType.ONEHOT == "OneHotEncoder"

    def test_scaler_type_values(self):
        assert ScalerType.MINMAX == "MinMaxScaler"
        assert ScalerType.STANDARD == "StandardScaler"
        assert ScalerType.ROBUST == "RobustScaler"

    def test_imputer_type_values(self):
        assert ImputerType.SIMPLE == "Simple"


class TestConstantMaps:
    def test_metric_column_map_covers_all_metrics(self):
        for metric in ThresholdMetric:
            assert metric in METRIC_COLUMN_MAP

    def test_encoder_class_map_covers_all_encoders(self):
        for enc in EncoderType:
            assert enc in ENCODER_CLASS_MAP

    def test_scaler_class_map_covers_all_scalers(self):
        for scaler in ScalerType:
            assert scaler in SCALER_CLASS_MAP

    def test_imputer_class_map_covers_all_imputers(self):
        for imp in ImputerType:
            assert imp in IMPUTER_CLASS_MAP


class TestHelperFunctions:
    def test_get_metric_column_name(self):
        assert get_metric_column_name(ThresholdMetric.MSE) == "MSE_Recons_error"
        assert get_metric_column_name(ThresholdMetric.MAE) == "MAE_Recons_error"
        assert get_metric_column_name(ThresholdMetric.RMSE) == "RMSE_Recons_error"
        assert get_metric_column_name(ThresholdMetric.MAX_ERROR) == "Max_Recons_error"

    def test_parse_threshold_metric_valid(self):
        assert parse_threshold_metric("mse") == ThresholdMetric.MSE
        assert parse_threshold_metric("MAE") == ThresholdMetric.MAE
        assert parse_threshold_metric("Rmse") == ThresholdMetric.RMSE

    def test_parse_threshold_metric_invalid(self):
        with pytest.raises(ValueError, match="Unknown threshold metric"):
            parse_threshold_metric("invalid_metric")


class TestExceptions:
    def test_base_exception(self):
        with pytest.raises(SegmentAEError):
            raise SegmentAEError("base error")

    def test_clustering_error(self):
        err = ClusteringError("test")
        assert "Clustering Error" in str(err)

    def test_reconstruction_error(self):
        err = ReconstructionError("test")
        assert "Reconstruction Error" in str(err)

    def test_validation_error_with_suggestion(self):
        err = ValidationError("bad input", suggestion="fix it")
        assert "bad input" in str(err)
        assert "fix it" in str(err)

    def test_validation_error_without_suggestion(self):
        err = ValidationError("bad input")
        assert "bad input" in str(err)

    def test_model_not_fitted_error_default(self):
        err = ModelNotFittedError(component="TestModel")
        assert "TestModel" in str(err)
        assert "fitted" in str(err)

    def test_model_not_fitted_error_custom(self):
        err = ModelNotFittedError(message="custom message")
        assert "custom message" in str(err)

    def test_configuration_error_with_options(self):
        err = ConfigurationError("bad config", valid_options=["a", "b"])
        assert "bad config" in str(err)
        assert "a" in str(err)

    def test_autoencoder_error(self):
        err = AutoencoderError("test")
        assert "Autoencoder Error" in str(err)

    def test_all_exceptions_inherit_from_base(self):
        for exc_class in [
            ClusteringError,
            ReconstructionError,
            ValidationError,
            ModelNotFittedError,
            ConfigurationError,
            AutoencoderError,
        ]:
            assert issubclass(exc_class, SegmentAEError)


class TestEnumFormatting:
    """Enum members must render as their value on every Python version."""

    @pytest.mark.parametrize(
        "member,value",
        [
            (ClusterModel.KMEANS, "KMeans"),
            (ThresholdMetric.MAX_ERROR, "max_error"),
            (EncoderType.IFREQUENCY, "IFrequencyEncoder"),
            (ScalerType.ROBUST, "RobustScaler"),
            (ImputerType.SIMPLE, "Simple"),
            (PhaseType.TESTING, "testing"),
        ],
    )
    def test_str_and_format_return_value(self, member, value):
        assert str(member) == value
        assert f"{member}" == value
        assert "{}".format(member) == value

    def test_members_remain_enum_instances(self):
        assert isinstance(ClusterModel.KMEANS, ClusterModel)
        assert ClusterModel("KMeans") is ClusterModel.KMEANS


class TestPerformanceMetric:
    def test_values_are_result_columns(self):
        from segmentae.core.constants import PerformanceMetric

        assert [m.value for m in PerformanceMetric] == [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ]

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("f1_score", "F1 Score"),
            ("F1-Score", "F1 Score"),
            ("f1", "F1 Score"),
            ("Recall", "Recall"),
        ],
    )
    def test_parse_performance_metric(self, name, expected):
        from segmentae.core.constants import parse_performance_metric

        assert parse_performance_metric(name).value == expected

    def test_parse_performance_metric_invalid(self):
        from segmentae.core.constants import parse_performance_metric

        with pytest.raises(ValueError, match="Invalid performance metric"):
            parse_performance_metric("auc")


class TestParseThresholdMetricInputs:
    def test_enum_passthrough_and_whitespace(self):
        assert parse_threshold_metric(ThresholdMetric.MAE) is ThresholdMetric.MAE
        assert parse_threshold_metric("  MSE ") == ThresholdMetric.MSE


class TestExceptionHierarchy:
    def test_input_errors_are_value_errors(self):
        # Lets callers catch invalid input uniformly with Pydantic's ValueError subclasses
        assert issubclass(ValidationError, ValueError)
        assert issubclass(ConfigurationError, ValueError)

    def test_configuration_error_without_options(self):
        assert "Valid options" not in str(ConfigurationError("bad config"))


class TestProtocols:
    def test_autoencoder_protocol_is_runtime_checkable(self):
        from segmentae.core.types import AutoencoderProtocol

        class WithPredict:
            def predict(self, X):
                return X

        assert isinstance(WithPredict(), AutoencoderProtocol)
        assert not isinstance(object(), AutoencoderProtocol)
