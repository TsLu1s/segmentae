import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from segmentae import SegmentAE, Clustering
from segmentae.processing.preprocessing import Preprocessing
from segmentae.autoencoders import DenseAutoencoder #, BatchNormAutoencoder
from segmentae.data_sources import load_dataset
from segmentae.metrics import metrics_classification

############################################################################################
### Data Loading

train, test, target = load_dataset(
    dataset_selection='htru2_dataset',  # Other Options Available
    split_ratio=0.75                         
)

test, future_data = train_test_split(test, train_size=0.9, random_state=5)

# Resetting Index
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)

X_train = train.drop(columns=[target]).copy()
X_test, y_test = test.drop(columns=[target]).copy(), test[target].astype(int)
X_future_data = future_data.drop(columns=[target]).copy()
y_future_data = future_data[target].astype(int)

############################################################################################
### Preprocessing with Enhanced Validation
pr = Preprocessing(
    encoder="IFrequencyEncoder",   # "LabelEncoder", "OneHotEncoder"
    scaler="StandardScaler",       # "MinMaxScaler", "RobustScaler"
    imputer=None
)

pr.fit(X=X_train)
X_train = pr.transform(X=X_train)
X_test = pr.transform(X=X_test)
X_future_data = pr.transform(X=X_future_data)

############################################################################################
### Clustering Implementation with Registry Pattern
cl_model = Clustering(
    cluster_model=["KMeans"],  # "KMeans", "MiniBatchKMeans", "GMM", "Agglomerative"
    n_clusters=3
)
cl_model.clustering_fit(X=X_train)

############################################################################################
### Built-in Autoencoder Implementation
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# DenseAutoencoder with comprehensive configuration
denseAutoencoder = DenseAutoencoder(
    hidden_dims=[16, 8, 4],           # Architecture definition
    encoder_activation='relu',
    decoder_activation='relu',
    optimizer='adam',
    learning_rate=0.001,
    epochs=150,
    val_size=0.15,
    stopping_patient=10,              # Early stopping
    dropout_rate=0.1,                 # Regularization
    batch_size=None
)

# Fit the autoencoder
denseAutoencoder.fit(input_data=X_train)

# Model inspection methods
denseAutoencoder.summary()            # Model architecture
denseAutoencoder.plot_training_loss() # Training visualization

# Alternative: BatchNormAutoencoder for better convergence
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
### Autoencoder + Clustering Integration

sg = SegmentAE(
    ae_model=denseAutoencoder,
    cl_model=cl_model
)

### Train Reconstruction with Metrics
sg.reconstruction(
    input_data=X_train,
    threshold_metric="mse" # || "mse", "mae", "rmse", "max_error"   Enum prevents typos
)

### Reconstruction Error Result Performance
results = sg.evaluation(
    input_data=X_test,
    target_col=y_test,
    threshold_ratio=0.75 # You can adjust the threshold ratio as needed, it multiples the default reconstruction metric by cluster
)                        

# Access detailed metadata
preds_test = sg.preds_test
recon_metrics_test = sg.reconstruction_test

print(f"Evaluation Results:\n{results['global metrics']}")

############################################################################################
### Multiple Threshold Ratio Evaluation with Enhanced Metrics
threshold_ratios = [0.75, 1, 1.5, 2, 3, 4]

global_results = pd.concat([
    sg.evaluation(
        input_data=X_test,
        target_col=y_test,
        threshold_ratio=thr
    )["global metrics"]
    for thr in threshold_ratios
])

print("\nThreshold Optimization Results:")
print(global_results)

############################################################################################
### Anomaly Detection Predictions
best_ratio = global_results.sort_values(by="Accuracy", ascending=False).iloc[0]["Threshold Ratio"]

predictions = sg.detections(
    input_data=X_future_data,
    threshold_ratio=best_ratio
)

# Use the new metrics module for evaluation
final_metrics = metrics_classification(
    y_true=y_future_data,
    y_pred=predictions['Predicted Anomalies']
)

print("\nFinal Performance Metrics:")
print(f"Accuracy: {final_metrics['Accuracy']}")
print(f"Precision: {final_metrics['Precision']}")
print(f"Recall: {final_metrics['Recall']}")
print(f"F1 Score: {final_metrics['F1 Score']}")











































