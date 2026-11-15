"""
SegmentAE Example: Grid Search Optimization
===========================================

This example demonstrates how to use SegmentAE_Optimizer to search over
autoencoders, clustering algorithms, numbers of clusters and threshold ratios.

The labeled data is split in two: a validation set, used by the optimizer to
select the best configuration, and a hold-out set that the search never sees,
used to report the final performance.
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

from segmentae.autoencoders import EnsembleAutoencoder
from segmentae.data_sources import load_dataset
from segmentae.metrics import metrics_classification
from segmentae.optimization import SegmentAE_Optimizer
from segmentae.processing.preprocessing import Preprocessing

############################################################################################
### Data Loading

train, test, target = load_dataset(
    dataset_selection="htru2_dataset",  # Options: 'default_credit_card', 'htru2_dataset', 'shuttle_148'
    split_ratio=0.75,
)

# Validation set: selects the configuration || Hold-out set: final, unbiased evaluation
valid, holdout = train_test_split(test, train_size=0.6, random_state=5, stratify=test[target])

X_train = train.drop(columns=[target]).copy()
X_valid, y_valid = valid.drop(columns=[target]).copy(), valid[target].astype(int)
X_holdout, y_holdout = holdout.drop(columns=[target]).copy(), holdout[target].astype(int)

############################################################################################
### Preprocessing with Type Safety and Validation

# MinMaxScaler maps the features to [0, 1], the range a sigmoid output layer can
# reconstruct. With "StandardScaler" or "RobustScaler", use linear output layers.
pr = Preprocessing(
    encoder="IFrequencyEncoder",  # "LabelEncoder", "OneHotEncoder"
    scaler="MinMaxScaler",  # "StandardScaler", "RobustScaler"
    imputer=None,
)

pr.fit(X=X_train)
X_train = pr.transform(X=X_train)
X_valid = pr.transform(X=X_valid)
X_holdout = pr.transform(X=X_holdout)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

############################################################################################
### AutoEncoder Model: Ensemble Autoencoder

# EnsembleAutoencoder combines multiple architectures for robustness
ensembleAutoencoder = EnsembleAutoencoder(
    n_autoencoders=3,
    hidden_dims=[[24, 12, 8, 4], [64, 32, 16, 4], [20, 10, 5]],
    encoder_activations=["relu", "tanh", "relu"],
    decoder_activations=["relu", "tanh", "relu"],
    optimizers=["adam", "sgd", "rmsprop"],
    learning_rates=[0.001, 0.01, 0.005],
    epochs_list=[10, 15, 20],
    val_size_list=[0.15, 0.15, 0.15],
    stopping_patients=[30, 30, 15],
    dropout_rates=[0.1, 0.2, 0.3],
    batch_sizes=[None, None, 128],
    use_batch_norm=[False, False, False],
    random_state=42,  # Reproducible training
)

# Train the ensemble
ensembleAutoencoder.fit(X_train)
print("Ensemble autoencoder trained successfully!")

############################################################################################
### AutoEncoder Model: Custom Basic Model

# Define encoder architecture
encoder = Sequential(
    [
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(24, activation="relu"),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
    ]
)

# Define decoder architecture
decoder = Sequential(
    [
        Input(shape=(8,)),
        Dense(16, activation="relu"),
        Dense(24, activation="relu"),
        Dense(32, activation="relu"),
        Dense(X_train.shape[1], activation="sigmoid"),
    ]
)

# Combine into autoencoder
basicAutoencoder = Sequential([encoder, decoder])
basicAutoencoder.compile(optimizer="adam", loss="mse")

# Train the model
basicAutoencoder.fit(X_train, X_train, epochs=50, batch_size=None, validation_split=0.1)
print("Basic autoencoder trained successfully!")

############################################################################################
### SegmentAE Optimizer Implementation with Enhanced Features

# Using string-based configuration (backward compatible)
optimizer = SegmentAE_Optimizer(
    autoencoder_models=[ensembleAutoencoder, basicAutoencoder],
    n_clusters_list=[1, 2, 3],
    cluster_models=["KMeans", "MiniBatchKMeans", "GMM"],
    threshold_ratios=[1, 1.5, 2, 4],
    performance_metric="Accuracy",
    verbose=True,  # Set to False to silence the progress output
)

# Using Enums as input parameters (recommended for new code)
# from segmentae.core import ClusterModel, PerformanceMetric
#
# optimizer = SegmentAE_Optimizer(
#     autoencoder_models=[ensembleAutoencoder, basicAutoencoder],
#     n_clusters_list=[1, 2, 3],
#     cluster_models=[ClusterModel.KMEANS, ClusterModel.MINIBATCH_KMEANS, ClusterModel.GMM],
#     threshold_ratios=[1, 1.5, 2, 4],
#     performance_metric=PerformanceMetric.ACCURACY
# )

# Run optimization: configurations are scored on the validation set
print("\nRunning grid search optimization...")
sg = optimizer.optimize(X_train, X_valid, y_valid)

# Access optimization results
preds_valid = sg.preds_test
recon_metrics_valid = sg.reconstruction_test
leaderboard = optimizer.leaderboard

print("\n" + "=" * 50)
print("OPTIMIZATION RESULTS")
print("=" * 50)
print(f"\nBest Validation Performance: {optimizer.best_performance:.4f}")
print("Best Configuration:")
print(f"  - Model: {optimizer.best_model}")
print(f"  - Clusters: {optimizer.best_n_clusters}")
print(f"  - Clustering Algorithm: {optimizer.best_cluster_model}")
print(f"  - Threshold Ratio: {optimizer.best_threshold_ratio}")
print("\nTop 10 Configurations:")
print(leaderboard.head(10).to_string())

############################################################################################
### Anomaly Detection with Optimal Configuration

threshold = optimizer.best_threshold_ratio

# Hold-out data was not used to select the configuration
predictions = sg.detections(input_data=X_holdout, threshold_ratio=threshold)

# Calculate final metrics
final_metrics = metrics_classification(y_holdout, predictions["Predicted Anomalies"])
print("\nFinal Hold-out Performance:")
for metric in final_metrics.columns:
    print(f"  {metric}: {final_metrics[metric].iloc[0]:.4f}")
