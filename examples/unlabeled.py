import tensorflow as tf

from segmentae import SegmentAE, Preprocessing, Clustering
from segmentae.autoencoders import DenseAutoencoder       #, BatchNormAutoencoder
from segmentae.data_sources import load_dataset


############################################################################################
### Data Loading (Unlabeled Scenario)

train, future_data, target = load_dataset(
    dataset_selection='htru2_dataset',  # Options: 'default_credit_card', 'htru2_dataset', 'shuttle_148'
    split_ratio=0.75                          
)

# Reset indices
train = train.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)

# For unlabeled scenario, we don't use the target column
X_train = train.drop(columns=[target]).copy()
X_future_data = future_data.drop(columns=[target]).copy()

# Save true labels for optional evaluation (in real scenario, these wouldn't exist)
# y_train = train[target].astype(int)
# y_future_data = future_data[target].astype(int)

############################################################################################
### Preprocessing with Enhanced Error Handling

pr = Preprocessing(
    encoder="IFrequencyEncoder",   # "LabelEncoder", "OneHotEncoder"
    scaler="StandardScaler",       # "MinMaxScaler", "RobustScaler"
    imputer=None
)

pr.fit(X=X_train)
X_train = pr.transform(X=X_train)
X_future_data = pr.transform(X=X_future_data)

############################################################################################

# GMM clustering for soft assignments
cl_model = Clustering(
    cluster_model=["KMeans"],  # "KMeans", "MiniBatchKMeans", "GMM", "Agglomerative"
    n_clusters=3,
    covariance_type='full'     # GMM-specific parameter
)

cl_model.clustering_fit(X=X_train)
print(f"Clustering completed: {cl_model.n_clusters} clusters identified")

############################################################################################
### Autoencoder Implementation for Unlabeled Data

print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

# DenseAutoencoder with architecture optimized for reconstruction
denseAutoencoder = DenseAutoencoder(
    hidden_dims=[16, 12, 8, 4],        # Bottleneck architecture
    encoder_activation='relu',
    decoder_activation='relu',
    optimizer='adam',
    learning_rate=0.001,
    epochs=150,
    val_size=0.15,                     # Validation split for monitoring
    stopping_patient=10,               # Early stopping
    dropout_rate=0.1,
    batch_size=None
)

# Train autoencoder on unlabeled data
denseAutoencoder.fit(input_data=X_train)
print("Autoencoder training completed")

# Display model information
denseAutoencoder.summary()
denseAutoencoder.plot_training_loss()

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

sg = SegmentAE(
    ae_model=denseAutoencoder,
    cl_model=cl_model
)

### Train Reconstruction (Unsupervised)
sg.reconstruction(
    input_data=X_train,
    threshold_metric="mse" # || "mse", "mae", "rmse", "max_error"  # Can also use MAE, RMSE, or MAX_ERROR
)

# Access reconstruction metadata
preds_train = sg.preds_train
recon_metrics_train = sg.reconstruction_eval

print("\n" + "="*50)
print("RECONSTRUCTION ANALYSIS")
print("="*50)
print(f"Reconstruction threshold: {sg.threshold:.6f}")
print(f"Mean reconstruction error: {recon_metrics_train['mse'].mean():.6f}")
print(f"Std reconstruction error: {recon_metrics_train['mse'].std():.6f}")

############################################################################################
### Anomaly Detection on Unlabeled Future Data
threshold_ratio = 2.0

predictions = sg.detections(
    input_data=X_future_data,
    threshold_ratio=threshold_ratio
)

# Analyze Detection Results
n_anomalies = predictions['Predicted Anomalies'].sum()
anomaly_rate = n_anomalies / len(predictions)

print("\n" + "="*50)
print("ANOMALY DETECTION RESULTS")
print("="*50)
print(f"Threshold ratio: {threshold_ratio}")
print(f"Anomalies detected: {n_anomalies} / {len(predictions)}")
print(f"Anomaly rate: {anomaly_rate:.2%}")

# Cluster-wise analysis
cluster_analysis = predictions.groupby('Cluster')['Predicted Anomalies'].agg([
    'count', 'sum', 'mean'
])
cluster_analysis.columns = ['Total Samples', 'Anomalies', 'Anomaly Rate']
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
for metric, value in actual_metrics.items():
    print(f"  {metric}: {value:.4f}")
"""

############################################################################################
############################################################################################









