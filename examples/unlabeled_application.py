"""
SegmentAE Example: Unlabeled Anomaly Detection
==============================================

This example demonstrates how to use SegmentAE for unsupervised anomaly detection
on unlabeled data, simulating real-world scenarios where ground truth is unavailable.

Features demonstrated:
- Unsupervised anomaly detection workflow
- Built-in DenseAutoencoder for reconstruction
- Reconstruction error analysis
- Cluster-wise anomaly detection
- Optional ground truth evaluation
"""

import pandas as pd
import tensorflow as tf

from segmentae import Clustering, Preprocessing, SegmentAE
from segmentae.autoencoders import DenseAutoencoder  # , BatchNormAutoencoder
from segmentae.data_sources import load_dataset

############################################################################################
### Data Loading (Unlabeled Scenario)

train, future_data, target = load_dataset(
    dataset_selection="htru2_dataset",  # Options: 'default_credit_card', 'htru2_dataset', 'shuttle_148'
    split_ratio=0.75,
)

# For unlabeled scenario, it is not used the target column
X_train = train.drop(columns=[target]).copy()
X_future_data = future_data.drop(columns=[target]).copy()

# Save true labels for optional evaluation (in real scenario, these wouldn't exist)
# y_train = train[target].astype(int)
# y_future_data = future_data[target].astype(int)

############################################################################################
### Preprocessing with Enhanced Error Handling

# MinMaxScaler maps the features to [0, 1], the range a sigmoid output layer can
# reconstruct. With "StandardScaler" or "RobustScaler", build the autoencoder
# with output_activation="linear" instead.
pr = Preprocessing(
    encoder="IFrequencyEncoder",  # "LabelEncoder", "OneHotEncoder"
    scaler="MinMaxScaler",  # "StandardScaler", "RobustScaler"
    imputer=None,
)

pr.fit(X=X_train)
X_train = pr.transform(X=X_train)
X_future_data = pr.transform(X=X_future_data)

############################################################################################
### Clustering Implementation

# KMeans clustering for hard assignments
cl_model = Clustering(
    cluster_model=["KMeans"],  # "KMeans", "MiniBatchKMeans", "GMM"
    n_clusters=3,
    covariance_type="full",  # GMM-specific parameter
)

cl_model.clustering_fit(X=X_train)
print(f"Clustering completed: {cl_model.n_clusters} clusters identified")

############################################################################################
### Autoencoder Implementation for Unlabeled Data

print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

denseAutoencoder = DenseAutoencoder(
    hidden_dims=[16, 12, 8, 4],  # Bottleneck architecture
    encoder_activation="relu",
    decoder_activation="relu",
    output_activation="sigmoid",  # matches MinMax-scaled data
    optimizer="adam",
    learning_rate=0.001,
    epochs=150,
    val_size=0.15,  # Validation split for monitoring
    stopping_patient=10,  # Early stopping
    dropout_rate=0.1,
    batch_size=None,
    random_state=42,
)

denseAutoencoder.fit(input_data=X_train)
print("Autoencoder training completed")

denseAutoencoder.summary()
# Training visualization (opt-in): opens a window that pauses the script until it is closed
# denseAutoencoder.plot_training_loss()

# Alternative: BatchNormAutoencoder for improved stability
"""
batchAutoencoder = BatchNormAutoencoder(
    hidden_dims=[32, 16, 8],
    encoder_activation='relu',
    decoder_activation='relu',
    optimizer='adam',
    learning_rate=0.001,
    epochs=150,
    val_size=0.15,
    stopping_patient=10,
    dropout_rate=0.1,
    batch_size=None
)
batchAutoencoder.fit(input_data=X_train)
"""

############################################################################################
### SegmentAE Integration for Unlabeled Anomaly Detection

sg = SegmentAE(ae_model=denseAutoencoder, cl_model=cl_model)

### Train Reconstruction (Unsupervised)

sg.reconstruction(
    input_data=X_train,
    threshold_metric="mse",  # "mse", "mae", "rmse", "max_error"
)

# Access reconstruction metadata (dict keyed by cluster id)
preds_train = sg.preds_train
recon_metrics_train = sg.reconstruction_eval

# Aggregate per-row reconstruction errors across all clusters
metric_col = "MSE_Recons_error"  # matches threshold_metric="mse"
all_errors = pd.concat(
    [recon_metrics_train[cluster_id]["metrics"][metric_col] for cluster_id in recon_metrics_train]
)

print("\n" + "=" * 50)
print("RECONSTRUCTION ANALYSIS")
print("=" * 50)
print(f"Mean reconstruction error: {all_errors.mean():.6f}")
print(f"Std reconstruction error: {all_errors.std():.6f}")
print("Baseline threshold per cluster (mean training error):")
for cluster_id, threshold in sg.thresholds.items():
    print(f"  Cluster {cluster_id}: {threshold:.6f}")

############################################################################################
### Anomaly Detection on Unlabeled Future Data

threshold_ratio = 2.0

predictions = sg.detections(input_data=X_future_data, threshold_ratio=threshold_ratio)

# Analyze Detection Results
n_anomalies = predictions["Predicted Anomalies"].sum()
anomaly_rate = n_anomalies / len(predictions)

print("\n" + "=" * 50)
print("ANOMALY DETECTION RESULTS")
print("=" * 50)
print(f"Threshold ratio: {threshold_ratio}")
print(f"Anomalies detected: {n_anomalies} / {len(predictions)}")
print(f"Anomaly rate: {anomaly_rate:.2%}")

# Every row comes with its anomaly score and the threshold it was compared against,
# so the most anomalous samples can be ranked without labels
print("\nMost anomalous samples (score relative to the cluster threshold):")
ranked = predictions.assign(
    Score_Ratio=predictions["Reconstruction Error"] / predictions["Threshold"]
).sort_values("Score_Ratio", ascending=False)
print(ranked[["Cluster", "Reconstruction Error", "Threshold", "Score_Ratio"]].head())

# Cluster-wise analysis
cluster_analysis = predictions.groupby("Cluster")["Predicted Anomalies"].agg(
    ["count", "sum", "mean"]
)
cluster_analysis.columns = ["Total Samples", "Anomalies", "Anomaly Rate"]
print("\nCluster-wise Analysis:")
print(cluster_analysis)

############################################################################################
### Optional: If ground truth is available for evaluation

"""
from segmentae.metrics import metrics_classification

actual_metrics = metrics_classification(
    y_true=y_future_data,
    y_pred=predictions['Predicted Anomalies']
)
print("\nActual Performance (if ground truth available):")
for metric in actual_metrics.columns:
    print(f"  {metric}: {actual_metrics[metric].iloc[0]:.4f}")
"""

############################################################################################
