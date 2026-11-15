"""Tests for autoencoder module."""

import numpy as np
import pytest

from segmentae.autoencoders.batch_norm import BatchNormAutoencoder
from segmentae.autoencoders.dense import DenseAutoencoder
from segmentae.autoencoders.ensemble import EnsembleAutoencoder
from segmentae.core.exceptions import ConfigurationError, ModelNotFittedError, ValidationError


class TestDenseAutoencoder:
    def test_init_defaults(self):
        ae = DenseAutoencoder()
        assert ae.hidden_dims == [12, 8, 4]
        assert ae.encoder_activation == "relu"
        assert ae.decoder_activation == "relu"
        assert ae.optimizer == "adam"
        assert ae.learning_rate == 0.001
        assert ae.autoencoder is None

    def test_fit_creates_model(self, scaled_data):
        ae = DenseAutoencoder(
            hidden_dims=[8, 4],
            epochs=5,
            val_size=0.2,
            stopping_patient=3,
        )
        ae.fit(scaled_data)

        assert ae.autoencoder is not None
        assert ae.input_dim == scaled_data.shape[1]
        assert ae.history is not None

    def test_predict_shape(self, scaled_data):
        ae = DenseAutoencoder(hidden_dims=[8, 4], epochs=5, stopping_patient=3)
        ae.fit(scaled_data)
        predictions = ae.predict(scaled_data)

        assert predictions.shape == scaled_data.shape

    def test_evaluate_returns_loss(self, scaled_data):
        ae = DenseAutoencoder(hidden_dims=[8, 4], epochs=5, stopping_patient=3)
        ae.fit(scaled_data)
        loss = ae.evaluate(scaled_data)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_invalid_optimizer_raises(self, scaled_data):
        ae = DenseAutoencoder(hidden_dims=[4], optimizer="invalid_opt", epochs=1)
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            ae.fit(scaled_data)

    def test_all_valid_optimizers(self, scaled_data):
        for opt in ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]:
            ae = DenseAutoencoder(hidden_dims=[4], optimizer=opt, epochs=1, stopping_patient=1)
            ae.fit(scaled_data)
            assert ae.autoencoder is not None

    def test_dropout(self, scaled_data):
        ae = DenseAutoencoder(hidden_dims=[8, 4], dropout_rate=0.3, epochs=3, stopping_patient=2)
        ae.fit(scaled_data)
        assert ae.autoencoder is not None

    def test_custom_batch_size(self, scaled_data):
        ae = DenseAutoencoder(hidden_dims=[8, 4], batch_size=16, epochs=3, stopping_patient=2)
        ae.fit(scaled_data)
        assert ae.autoencoder is not None


class TestBatchNormAutoencoder:
    def test_init_defaults(self):
        ae = BatchNormAutoencoder()
        assert ae.hidden_dims == [32, 16, 8]
        assert ae.autoencoder is None

    def test_fit_creates_model(self, scaled_data):
        ae = BatchNormAutoencoder(hidden_dims=[8, 4], epochs=5, stopping_patient=3)
        ae.fit(scaled_data)

        assert ae.autoencoder is not None
        assert ae.input_dim == scaled_data.shape[1]

    def test_predict_shape(self, scaled_data):
        ae = BatchNormAutoencoder(hidden_dims=[8, 4], epochs=5, stopping_patient=3)
        ae.fit(scaled_data)
        predictions = ae.predict(scaled_data)
        assert predictions.shape == scaled_data.shape

    def test_evaluate(self, scaled_data):
        ae = BatchNormAutoencoder(hidden_dims=[8, 4], epochs=5, stopping_patient=3)
        ae.fit(scaled_data)
        loss = ae.evaluate(scaled_data)
        assert isinstance(loss, float)
        assert loss >= 0


class TestEnsembleAutoencoder:
    def test_init_defaults(self):
        ae = EnsembleAutoencoder()
        assert ae.n_autoencoders == 3
        assert len(ae.hidden_dims) == 3
        assert len(ae.autoencoders) == 0

    def test_fit_creates_multiple_models(self, scaled_data):
        ae = EnsembleAutoencoder(
            n_autoencoders=2,
            hidden_dims=[[8, 4]] * 2,
            encoder_activations=["relu"] * 2,
            decoder_activations=["relu"] * 2,
            optimizers=["adam"] * 2,
            learning_rates=[0.001] * 2,
            epochs_list=[3] * 2,
            val_size_list=[0.2] * 2,
            stopping_patients=[2] * 2,
            dropout_rates=[0.0] * 2,
            batch_sizes=[32] * 2,
            use_batch_norm=[False] * 2,
        )
        ae.fit(scaled_data)
        assert len(ae.autoencoders) == 2

    def test_predict_shape(self, scaled_data):
        ae = EnsembleAutoencoder(
            n_autoencoders=2,
            hidden_dims=[[8, 4]] * 2,
            encoder_activations=["relu"] * 2,
            decoder_activations=["relu"] * 2,
            optimizers=["adam"] * 2,
            learning_rates=[0.001] * 2,
            epochs_list=[3] * 2,
            val_size_list=[0.2] * 2,
            stopping_patients=[2] * 2,
            dropout_rates=[0.0] * 2,
            batch_sizes=[32] * 2,
            use_batch_norm=[False] * 2,
        )
        ae.fit(scaled_data)
        predictions = ae.predict(scaled_data)

        assert predictions.shape == scaled_data.shape

    def test_with_batch_norm(self, scaled_data):
        ae = EnsembleAutoencoder(
            n_autoencoders=2,
            hidden_dims=[[8, 4]] * 2,
            encoder_activations=["relu"] * 2,
            decoder_activations=["relu"] * 2,
            optimizers=["adam"] * 2,
            learning_rates=[0.001] * 2,
            epochs_list=[3] * 2,
            val_size_list=[0.2] * 2,
            stopping_patients=[2] * 2,
            dropout_rates=[0.0] * 2,
            batch_sizes=[32] * 2,
            use_batch_norm=[True] * 2,
        )
        ae.fit(scaled_data)
        assert len(ae.autoencoders) == 2

    def test_mismatched_param_lengths_raises(self):
        # Validation must not rely on ``assert`` (stripped under ``python -O``)
        with pytest.raises(ConfigurationError, match="same length as n_autoencoders"):
            EnsembleAutoencoder(
                n_autoencoders=2,
                hidden_dims=[[8, 4]],  # Only 1 instead of 2
            )

    def test_evaluate(self, scaled_data):
        ae = EnsembleAutoencoder(
            n_autoencoders=2,
            hidden_dims=[[8, 4]] * 2,
            encoder_activations=["relu"] * 2,
            decoder_activations=["relu"] * 2,
            optimizers=["adam"] * 2,
            learning_rates=[0.001] * 2,
            epochs_list=[3] * 2,
            val_size_list=[0.2] * 2,
            stopping_patients=[2] * 2,
            dropout_rates=[0.0] * 2,
            batch_sizes=[32] * 2,
            use_batch_norm=[False] * 2,
        )
        ae.fit(scaled_data)
        results = ae.evaluate(scaled_data)
        assert len(results) == 2
        assert all(r >= 0 for r in results)


def _small_ensemble(**overrides):
    params = dict(n_autoencoders=2, hidden_dims=[[4, 2]] * 2, epochs_list=[1, 1], verbose=0)
    params.update(overrides)
    return EnsembleAutoencoder(**params)


class TestUnfittedAutoencoders:
    @pytest.mark.parametrize("cls", [DenseAutoencoder, BatchNormAutoencoder, EnsembleAutoencoder])
    def test_predict_before_fit_raises(self, cls, scaled_data):
        with pytest.raises(ModelNotFittedError, match="must be trained"):
            cls().predict(scaled_data)

    @pytest.mark.parametrize("cls", [DenseAutoencoder, BatchNormAutoencoder, EnsembleAutoencoder])
    def test_evaluate_before_fit_raises(self, cls, scaled_data):
        with pytest.raises(ModelNotFittedError):
            cls().evaluate(scaled_data)

    @pytest.mark.parametrize("cls", [DenseAutoencoder, BatchNormAutoencoder, EnsembleAutoencoder])
    def test_save_before_fit_raises(self, cls, tmp_path):
        with pytest.raises(ModelNotFittedError):
            cls().save_model(str(tmp_path / "model.keras"))

    @pytest.mark.parametrize("cls", [DenseAutoencoder, BatchNormAutoencoder, EnsembleAutoencoder])
    def test_summary_and_plot_before_fit_do_not_raise(self, cls, capsys):
        ae = cls()
        ae.summary()
        ae.plot_training_loss()
        assert "fit" in capsys.readouterr().out


class TestHyperparameterValidation:
    @pytest.mark.parametrize(
        "params",
        [
            {"hidden_dims": []},
            {"hidden_dims": [8, 0]},
            {"hidden_dims": "8,4"},
            {"dropout_rate": 1.0},
            {"dropout_rate": -0.1},
            {"learning_rate": 0},
            {"epochs": 0},
            {"val_size": 1.0},
            {"stopping_patient": -1},
            {"batch_size": 0},
            {"optimizer": "not_an_optimizer"},
        ],
    )
    def test_invalid_hyperparameters_raise_configuration_error(self, params, scaled_data):
        ae = DenseAutoencoder(**params)
        with pytest.raises(ConfigurationError):
            ae.fit(scaled_data)
        assert ae.autoencoder is None  # nothing was built

    def test_ensemble_member_hyperparameters_are_validated(self, scaled_data):
        ae = _small_ensemble(dropout_rates=[0.1, 1.5])
        with pytest.raises(ConfigurationError, match="Autoencoder 2"):
            ae.fit(scaled_data)

    def test_ensemble_invalid_n_autoencoders(self):
        with pytest.raises(ConfigurationError):
            EnsembleAutoencoder(n_autoencoders=0)

    def test_ensemble_defaults_scale_with_n_autoencoders(self):
        ae = EnsembleAutoencoder(n_autoencoders=5)
        assert len(ae.hidden_dims) == len(ae.optimizers) == len(ae.output_activations) == 5
        ae.hidden_dims[0].append(99)  # members must not share one default list
        assert ae.hidden_dims[1] == [12, 8, 4]


class TestTrainingData:
    def test_nan_training_data_raises(self, scaled_data):
        data = scaled_data.copy()
        data.iloc[0, 0] = np.nan
        with pytest.raises(ValidationError, match="missing or infinite"):
            DenseAutoencoder(hidden_dims=[4], epochs=1).fit(data)

    def test_non_numeric_training_data_raises(self, sample_mixed_data):
        with pytest.raises(ValidationError, match="numeric"):
            DenseAutoencoder(hidden_dims=[4], epochs=1).fit(sample_mixed_data)

    def test_numpy_input_is_accepted(self, scaled_data):
        ae = DenseAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0)
        ae.fit(scaled_data.to_numpy())
        assert ae.predict(scaled_data.to_numpy()).shape == scaled_data.shape

    def test_predict_with_wrong_number_of_features_raises(self, scaled_data):
        ae = DenseAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0)
        ae.fit(scaled_data)
        with pytest.raises(ValidationError, match="features"):
            ae.predict(scaled_data.iloc[:, :2])

    def test_sigmoid_output_warns_on_out_of_range_data(self, scaled_data):
        standardized = scaled_data * 6 - 3
        with pytest.warns(UserWarning, match="sigmoid"):
            DenseAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0).fit(standardized)

    def test_linear_output_reconstructs_negative_values_without_warning(self, scaled_data, recwarn):
        standardized = scaled_data * 6 - 3
        ae = DenseAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0, output_activation="linear")
        ae.fit(standardized)
        assert not [w for w in recwarn if "output layer" in str(w.message)]
        assert ae.predict(standardized).min() < 0

    def test_wide_layers_warn(self, scaled_data):
        with pytest.warns(UserWarning, match="overfitting"):
            DenseAutoencoder(hidden_dims=[64, 4], epochs=1, verbose=0).fit(scaled_data)


class TestTrainingBehaviour:
    def test_random_state_makes_training_reproducible(self, scaled_data):
        first = DenseAutoencoder(hidden_dims=[4, 2], epochs=2, verbose=0, random_state=7)
        second = DenseAutoencoder(hidden_dims=[4, 2], epochs=2, verbose=0, random_state=7)
        first.fit(scaled_data)
        second.fit(scaled_data)
        np.testing.assert_allclose(first.predict(scaled_data), second.predict(scaled_data))

    def test_training_without_validation_split(self, scaled_data, recwarn):
        ae = DenseAutoencoder(hidden_dims=[4, 2], epochs=2, val_size=0, verbose=0)
        ae.fit(scaled_data)
        assert "val_loss" not in ae.history.history
        assert not [w for w in recwarn if "Early stopping" in str(w.message)]

    def test_predict_and_evaluate_are_quiet_by_default(self, scaled_data, capsys):
        ae = BatchNormAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0)
        ae.fit(scaled_data)
        capsys.readouterr()
        ae.predict(scaled_data)
        ae.evaluate(scaled_data)
        assert capsys.readouterr().out == ""

    def test_batch_norm_only_in_batch_norm_autoencoder(self, scaled_data):
        dense = DenseAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0)
        batch = BatchNormAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0)
        dense.fit(scaled_data)
        batch.fit(scaled_data)
        names = lambda ae: [type(layer).__name__ for layer in ae.autoencoder.layers]
        assert "BatchNormalization" not in names(dense)
        assert names(batch).count("BatchNormalization") == 3  # 2 encoder + 1 decoder layers

    def test_save_model_round_trip(self, scaled_data, tmp_path):
        import keras

        ae = DenseAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0)
        ae.fit(scaled_data)
        path = str(tmp_path / "dense.keras")
        ae.save_model(path)
        restored = keras.models.load_model(path)
        np.testing.assert_allclose(
            restored.predict(scaled_data.to_numpy(dtype="float32"), verbose=0),
            ae.predict(scaled_data),
            rtol=1e-5,
            atol=1e-6,
        )


class TestEnsembleBehaviour:
    def test_refit_replaces_members(self, scaled_data):
        ae = _small_ensemble()
        ae.fit(scaled_data)
        ae.fit(scaled_data)
        assert len(ae.autoencoders) == 2
        assert len(ae.histories) == 2
        assert ae.predict(scaled_data).shape == scaled_data.shape

    def test_fit_returns_self(self, scaled_data):
        ae = _small_ensemble()
        assert ae.fit(scaled_data) is ae

    def test_prediction_is_mean_of_members(self, scaled_data):
        ae = _small_ensemble()
        ae.fit(scaled_data)
        data = scaled_data.to_numpy(dtype="float32")
        expected = np.mean([m.predict(data, verbose=0) for m in ae.autoencoders], axis=0)
        np.testing.assert_allclose(ae.predict(scaled_data), expected, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize(
        "file_format,extension", [(None, ".h5"), ("h5", ".h5"), ("keras", ".keras")]
    )
    def test_save_model_writes_one_file_per_member(
        self, scaled_data, tmp_path, file_format, extension
    ):
        import os

        ae = _small_ensemble()
        ae.fit(scaled_data)
        prefix = str(tmp_path / "ensemble")
        saved = ae.save_model(prefix) if file_format is None else ae.save_model(prefix, file_format)

        # The default keeps the file names of earlier versions (.h5)
        assert saved == [f"{prefix}_autoencoder_{i}{extension}" for i in (1, 2)]
        assert all(os.path.exists(path) for path in saved)

    def test_save_model_rejects_unknown_format(self, scaled_data, tmp_path):
        ae = _small_ensemble()
        ae.fit(scaled_data)
        with pytest.raises(ConfigurationError):
            ae.save_model(str(tmp_path / "ensemble"), file_format="pickle")


class TestLearningRateSchedule:
    def test_keras_schedule_is_accepted(self, scaled_data):
        import keras

        schedule = keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=10, decay_rate=0.9)
        ae = DenseAutoencoder(hidden_dims=[4, 2], epochs=1, verbose=0, learning_rate=schedule)
        ae.fit(scaled_data)
        assert ae.predict(scaled_data).shape == scaled_data.shape

    @pytest.mark.parametrize("learning_rate", [0, -0.01, True])
    def test_non_positive_number_is_rejected(self, scaled_data, learning_rate):
        with pytest.raises(ConfigurationError, match="learning_rate"):
            DenseAutoencoder(hidden_dims=[4], epochs=1, learning_rate=learning_rate).fit(
                scaled_data
            )
