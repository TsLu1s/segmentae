"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from segmentae.core.constants import EncoderType, ScalerType
from segmentae.core.exceptions import ModelNotFittedError, ValidationError
from segmentae.processing.preprocessing import Preprocessing, PreprocessingConfig


class TestPreprocessingConfig:
    def test_default_config(self):
        config = PreprocessingConfig()
        assert config.encoder is None
        assert config.scaler == ScalerType.MINMAX
        assert config.imputer is not None

    def test_string_encoder(self):
        config = PreprocessingConfig(encoder="IFrequencyEncoder")
        assert config.encoder == EncoderType.IFREQUENCY

    def test_enum_encoder(self):
        config = PreprocessingConfig(encoder=EncoderType.LABEL)
        assert config.encoder == EncoderType.LABEL

    def test_none_encoder(self):
        config = PreprocessingConfig(encoder=None)
        assert config.encoder is None

    def test_invalid_encoder(self):
        with pytest.raises(PydanticValidationError):
            PreprocessingConfig(encoder="InvalidEncoder")

    def test_string_scaler(self):
        config = PreprocessingConfig(scaler="StandardScaler")
        assert config.scaler == ScalerType.STANDARD

    def test_none_scaler(self):
        config = PreprocessingConfig(scaler=None)
        assert config.scaler is None

    def test_invalid_scaler(self):
        with pytest.raises(PydanticValidationError):
            PreprocessingConfig(scaler="InvalidScaler")


class TestPreprocessing:
    def test_fit_numerical_only(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, scaler="MinMaxScaler", imputer=None)
        pr.fit(sample_numerical_data)
        assert pr._is_fitted
        assert len(pr.cat_cols) == 0
        assert len(pr.num_cols) == sample_numerical_data.shape[1]

    def test_transform_numerical(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, scaler="MinMaxScaler", imputer=None)
        pr.fit(sample_numerical_data)
        transformed = pr.transform(sample_numerical_data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_numerical_data.shape
        # MinMax should scale to [0, 1] range (approximately, due to float precision)
        assert transformed.min().min() >= -0.01
        assert transformed.max().max() <= 1.01

    def test_fit_transform(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, scaler="MinMaxScaler", imputer=None)
        transformed = pr.fit_transform(sample_numerical_data)
        assert isinstance(transformed, pd.DataFrame)
        assert pr._is_fitted

    def test_transform_before_fit_raises(self, sample_numerical_data):
        pr = Preprocessing()
        with pytest.raises(ModelNotFittedError):
            pr.transform(sample_numerical_data)

    def test_invalid_input_type(self):
        pr = Preprocessing()
        with pytest.raises(ValidationError):
            pr.fit(np.array([[1, 2], [3, 4]]))

    def test_empty_dataframe_raises(self):
        pr = Preprocessing()
        with pytest.raises(ValidationError):
            pr.fit(pd.DataFrame())

    def test_standard_scaler(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, scaler="StandardScaler", imputer=None)
        pr.fit(sample_numerical_data)
        transformed = pr.transform(sample_numerical_data)

        # Standard scaling: mean ~ 0, std ~ 1
        assert abs(transformed.mean().mean()) < 0.1
        assert abs(transformed.std().mean() - 1.0) < 0.1

    def test_robust_scaler(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, scaler="RobustScaler", imputer=None)
        pr.fit(sample_numerical_data)
        transformed = pr.transform(sample_numerical_data)
        assert transformed.shape == sample_numerical_data.shape

    def test_no_scaler(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, scaler=None, imputer=None)
        pr.fit(sample_numerical_data)
        transformed = pr.transform(sample_numerical_data)
        pd.testing.assert_frame_equal(transformed, sample_numerical_data)

    def test_mixed_data_with_encoder(self, sample_mixed_data):
        pr = Preprocessing(encoder="IFrequencyEncoder", scaler="MinMaxScaler", imputer=None)
        pr.fit(sample_mixed_data)
        transformed = pr.transform(sample_mixed_data)

        assert isinstance(transformed, pd.DataFrame)
        assert len(pr.cat_cols) == 2
        # All columns should be numerical after encoding
        assert transformed.select_dtypes(include=["object", "category"]).shape[1] == 0

    def test_repr(self):
        pr = Preprocessing(encoder="IFrequencyEncoder", scaler="MinMaxScaler", imputer=None)
        repr_str = repr(pr)
        assert "Preprocessing" in repr_str
        assert "IFrequencyEncoder" in repr_str

    def test_properties(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, scaler="MinMaxScaler", imputer=None)
        pr.fit(sample_numerical_data)
        assert pr.encoder is None
        assert pr.scaler is not None
        assert pr.imputer is None

    def test_imputer_with_nans(self, sample_data_with_nans):
        pr = Preprocessing(encoder=None, scaler="MinMaxScaler", imputer="Simple")
        pr.fit(sample_data_with_nans)
        transformed = pr.transform(sample_data_with_nans)
        assert transformed.isnull().sum().sum() == 0


class TestIndexSafety:
    """Transformations are positional: the input index never has to be reset."""

    @pytest.mark.parametrize("scaler", ["MinMaxScaler", "StandardScaler", "RobustScaler"])
    def test_non_default_index_gives_same_values(self, sample_numerical_data, scaler):
        pr = Preprocessing(encoder=None, scaler=scaler, imputer=None).fit(sample_numerical_data)
        reference = pr.transform(sample_numerical_data)

        shifted = sample_numerical_data.copy()
        shifted.index = shifted.index + 5000
        result = pr.transform(shifted)

        assert result.index.equals(shifted.index)
        assert result.isnull().sum().sum() == 0
        np.testing.assert_allclose(result.to_numpy(), reference.to_numpy())

    def test_split_without_index_reset(self, sample_mixed_data):
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(sample_mixed_data, train_size=0.7, random_state=0)
        pr = Preprocessing(encoder="IFrequencyEncoder", scaler="MinMaxScaler", imputer="Simple")
        transformed = pr.fit(train).transform(test)

        assert transformed.index.equals(test.index)
        assert transformed.isnull().sum().sum() == 0

    def test_imputer_with_non_default_index(self, sample_data_with_nans):
        data = sample_data_with_nans.copy()
        data.index = [f"row-{i}" for i in range(len(data))]
        transformed = Preprocessing(encoder=None, imputer="Simple").fit_transform(data)
        assert transformed.isnull().sum().sum() == 0
        assert list(transformed.index) == list(data.index)

    def test_input_is_not_modified(self, sample_data_with_nans):
        original = sample_data_with_nans.copy()
        Preprocessing(encoder=None, imputer="Simple").fit_transform(sample_data_with_nans)
        pd.testing.assert_frame_equal(sample_data_with_nans, original)


class TestColumnHandling:
    def test_missing_column_raises_validation_error(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, imputer=None).fit(sample_numerical_data)
        with pytest.raises(ValidationError, match="missing columns"):
            pr.transform(sample_numerical_data.drop(columns=["feature_2"]))

    def test_columns_returned_in_fitted_order(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, imputer=None).fit(sample_numerical_data)
        reference = pr.transform(sample_numerical_data)
        shuffled = sample_numerical_data[sample_numerical_data.columns[::-1]]
        pd.testing.assert_frame_equal(pr.transform(shuffled), reference)

    def test_unseen_column_passes_through_with_warning(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, imputer=None).fit(sample_numerical_data)
        extended = sample_numerical_data.assign(extra=100.0)
        with pytest.warns(UserWarning, match="not seen during fit"):
            transformed = pr.transform(extended)
        assert (transformed["extra"] == 100.0).all()
        assert list(transformed.columns) == list(sample_numerical_data.columns) + ["extra"]

    def test_duplicated_column_names_raise(self):
        data = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "a"])
        with pytest.raises(ValidationError, match="duplicated"):
            Preprocessing().fit(data)

    def test_feature_names(self, sample_mixed_data):
        pr = Preprocessing(encoder="OneHotEncoder", scaler="MinMaxScaler", imputer=None)
        transformed = pr.fit_transform(sample_mixed_data)
        assert pr.feature_names_in == list(sample_mixed_data.columns)
        assert pr.feature_names_out == list(transformed.columns)
        assert "cat_1_A" in pr.feature_names_out

    def test_all_numeric_widths_are_scaled(self):
        data = pd.DataFrame(
            {
                "f32": np.linspace(0, 100, 50).astype("float32"),
                "i16": np.arange(50, dtype="int16") * 10,
                "u8": np.arange(50, dtype="uint8"),
                "nullable": pd.array(np.arange(50) * 3, dtype="Int64"),
                "flag": [True, False] * 25,
            }
        )
        pr = Preprocessing(encoder=None, scaler="MinMaxScaler", imputer=None)
        transformed = pr.fit_transform(data)

        assert pr.num_cols == ["f32", "i16", "u8", "nullable"]  # booleans are left as they are
        for column in pr.num_cols:
            assert transformed[column].min() == pytest.approx(0.0)
            assert transformed[column].max() == pytest.approx(1.0)
        assert transformed["flag"].tolist() == data["flag"].tolist()

    def test_category_and_string_dtypes_are_encoded(self):
        data = pd.DataFrame(
            {
                "num": np.arange(20, dtype=float),
                "obj": ["a", "b"] * 10,
                "cat": pd.Categorical(["x", "y", "z", "x"] * 5),
                "txt": pd.array(["p", "q"] * 10, dtype="string"),
            }
        )
        pr = Preprocessing(encoder="IFrequencyEncoder", scaler="MinMaxScaler", imputer=None)
        transformed = pr.fit_transform(data)
        assert pr.cat_cols == ["obj", "cat", "txt"]
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in transformed.dtypes)


class TestImputation:
    def test_missing_values_only_at_inference_are_imputed(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, scaler="MinMaxScaler", imputer="Simple")
        pr.fit(sample_numerical_data)  # training data is complete
        assert pr.imputer is not None

        with_gap = sample_numerical_data.copy()
        with_gap.iloc[0, 0] = np.nan
        transformed = pr.transform(with_gap)

        assert transformed.isnull().sum().sum() == 0
        expected = pr.transform(sample_numerical_data)["feature_1"].mean()
        assert transformed.iloc[0, 0] == pytest.approx(expected)

    def test_imputed_value_is_the_training_mean(self):
        train = pd.DataFrame({"a": [0.0, 10.0, np.nan, 20.0]})
        pr = Preprocessing(encoder=None, scaler=None, imputer="Simple").fit(train)
        result = pr.transform(pd.DataFrame({"a": [np.nan, 5.0]}))
        assert result["a"].tolist() == [10.0, 5.0]

    def test_no_imputer_keeps_missing_values(self, sample_data_with_nans):
        pr = Preprocessing(encoder=None, scaler="MinMaxScaler", imputer=None)
        transformed = pr.fit_transform(sample_data_with_nans)
        assert transformed.isnull().sum().sum() == sample_data_with_nans.isnull().sum().sum()

    def test_complete_data_keeps_integer_dtype_without_scaler(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        transformed = Preprocessing(encoder=None, scaler=None, imputer="Simple").fit_transform(data)
        pd.testing.assert_frame_equal(transformed, data)

    def test_entirely_missing_column_raises(self):
        data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValidationError, match="no observed values"):
            Preprocessing(encoder=None, imputer="Simple").fit(data)

    def test_missing_categories_are_imputed_after_encoding(self):
        data = pd.DataFrame({"num": np.arange(30, dtype=float), "cat": ["a", "b", None] * 10})
        pr = Preprocessing(encoder="IFrequencyEncoder", scaler="MinMaxScaler", imputer="Simple")
        transformed = pr.fit_transform(data)
        assert transformed.isnull().sum().sum() == 0


class TestRefitAndState:
    def test_refit_resets_previous_components(self, sample_mixed_data, sample_numerical_data):
        pr = Preprocessing(encoder="IFrequencyEncoder", scaler="MinMaxScaler", imputer="Simple")
        pr.fit(sample_mixed_data)
        assert pr.encoder is not None

        pr.fit(sample_numerical_data)
        assert pr.encoder is None
        assert pr.cat_cols == []
        assert pr.feature_names_in == list(sample_numerical_data.columns)
        assert pr.transform(sample_numerical_data).shape == sample_numerical_data.shape

    def test_training_data_is_not_retained(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, imputer=None).fit(sample_numerical_data)
        held = [value for value in vars(pr).values() if isinstance(value, pd.DataFrame)]
        assert held == []

    def test_unencoded_categoricals_warn(self, sample_mixed_data):
        with pytest.warns(UserWarning, match="non-numeric"):
            Preprocessing(encoder=None, scaler="MinMaxScaler", imputer=None).fit(sample_mixed_data)

    def test_properties_return_copies(self, sample_numerical_data):
        pr = Preprocessing(encoder=None, imputer=None).fit(sample_numerical_data)
        pr.num_cols.append("tampered")
        assert "tampered" not in pr.num_cols
        assert pr.is_fitted

    @pytest.mark.parametrize("encoder", ["IFrequencyEncoder", "LabelEncoder", "OneHotEncoder"])
    def test_all_encoders_produce_numeric_output(self, sample_mixed_data, encoder):
        pr = Preprocessing(encoder=encoder, scaler="StandardScaler", imputer="Simple")
        transformed = pr.fit_transform(sample_mixed_data)
        assert len(transformed) == len(sample_mixed_data)
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in transformed.dtypes)
        assert transformed.isnull().sum().sum() == 0

    def test_enum_configuration(self, sample_mixed_data):
        from segmentae.core.constants import ImputerType

        pr = Preprocessing(
            encoder=EncoderType.LABEL, scaler=ScalerType.ROBUST, imputer=ImputerType.SIMPLE
        )
        assert (
            repr(pr) == "Preprocessing(encoder=LabelEncoder, scaler=RobustScaler, imputer=Simple)"
        )
        assert pr.fit_transform(sample_mixed_data).shape[0] == len(sample_mixed_data)

    def test_invalid_imputer(self):
        with pytest.raises(PydanticValidationError):
            Preprocessing(imputer="KNN")


class TestComponentFactory:
    def test_none_returns_none(self):
        from segmentae.processing.simplifier import ComponentFactory

        assert ComponentFactory.create_encoder(None) is None
        assert ComponentFactory.create_scaler(None) is None
        assert ComponentFactory.create_imputer(None) is None

    def test_unknown_type_raises_configuration_error(self):
        from segmentae.core.exceptions import ConfigurationError
        from segmentae.processing.simplifier import ComponentFactory

        with pytest.raises(ConfigurationError, match="Valid options"):
            ComponentFactory.create_scaler("NotAScaler")

    def test_unimportable_atlantic_gives_actionable_error(self, monkeypatch):
        """An installed Atlantic that cannot be imported must not be reported as missing."""
        from segmentae.core.exceptions import ConfigurationError
        from segmentae.processing import simplifier

        monkeypatch.setitem(simplifier._ATLANTIC_MODULES, "scaler", ("atlantic.no_such_module",))
        with pytest.raises(ConfigurationError, match="could not be imported") as error:
            simplifier.ComponentFactory.create_scaler("MinMaxScaler")
        assert "atlantic<2" in str(error.value)
        assert "not installed" not in str(error.value)

    @pytest.mark.parametrize("scaler", list(ScalerType))
    def test_scalers_created_from_enum_and_string(self, scaler):
        from segmentae.processing.simplifier import ComponentFactory

        assert type(ComponentFactory.create_scaler(scaler)).__name__ == f"Auto{scaler.value}"
        assert type(ComponentFactory.create_scaler(scaler.value)).__name__ == f"Auto{scaler.value}"


class TestCategoryDtypeEncoding:
    """Encoded ``category`` columns must come out numeric (they can stay categorical upstream)."""

    @pytest.fixture
    def category_data(self):
        # Every category has a different frequency, the case where a mapped
        # categorical column keeps its category dtype
        values = ["a"] * 30 + ["b"] * 20 + ["c"] * 10
        return pd.DataFrame({"num": np.linspace(0, 1, 60), "cat": pd.Categorical(values)}).sample(
            frac=1, random_state=0
        )

    @pytest.mark.parametrize("encoder", ["IFrequencyEncoder", "LabelEncoder", "OneHotEncoder"])
    def test_encoded_category_column_is_numeric(self, category_data, encoder):
        pr = Preprocessing(encoder=encoder, scaler="MinMaxScaler", imputer=None)
        transformed = pr.fit_transform(category_data)

        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in transformed.dtypes)
        assert transformed.min().min() >= 0 and transformed.max().max() <= 1  # scaled too

    def test_category_column_runs_through_clustering(self, category_data):
        from segmentae.clusters.clustering import Clustering

        transformed = Preprocessing(encoder="IFrequencyEncoder", imputer=None).fit_transform(
            category_data
        )
        clustering = Clustering(cluster_model=["KMeans"], n_clusters=2).clustering_fit(transformed)
        assert len(clustering.cluster_prediction(transformed)) == len(category_data)

    def test_same_values_as_string_column(self, category_data):
        as_text = category_data.assign(cat=category_data["cat"].astype(object))
        from_category = Preprocessing(encoder="IFrequencyEncoder", imputer=None).fit_transform(
            category_data
        )
        from_text = Preprocessing(encoder="IFrequencyEncoder", imputer=None).fit_transform(as_text)
        np.testing.assert_allclose(
            from_category.to_numpy(dtype=float), from_text.to_numpy(dtype=float)
        )


class TestUnseenColumnPlacement:
    def test_unseen_columns_come_last_with_one_hot_encoding(self, sample_mixed_data):
        pr = Preprocessing(encoder="OneHotEncoder", scaler="MinMaxScaler", imputer=None)
        pr.fit(sample_mixed_data)
        with pytest.warns(UserWarning, match="not seen during fit"):
            transformed = pr.transform(sample_mixed_data.assign(extra=1.0))
        assert list(transformed.columns) == pr.feature_names_out + ["extra"]
