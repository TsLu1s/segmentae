import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from segmentae.metrics import metrics_classification
from segmentae.processing.preprocessing import Preprocessing
from segmentae.autoencoders import EnsembleAutoencoder
from segmentae.optimization import SegmentAE_Optimizer
from segmentae.data_sources import load_dataset

############################################################################################
### Data Loading

train, test, target = load_dataset(
    dataset_selection='htru2_dataset',  # Options: 'default_credit_card', 'htru2_dataset', 'shuttle_148'
    split_ratio=0.75                          
)

X_train = train.drop(columns=[target]).copy()
X_test = test.drop(columns=[target]).copy()
y_test = test[target].astype(int)

############################################################################################
### Preprocessing with Type Safety and Validation

pr = Preprocessing(
    encoder="IFrequencyEncoder",   # "LabelEncoder", "OneHotEncoder"
    scaler="StandardScaler",       # "MinMaxScaler", "RobustScaler"
    imputer=None
)

pr.fit(X=X_train)
X_train = pr.transform(X=X_train)
X_test = pr.transform(X=X_test)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

############################################################################################
### AutoEncoder Model: Ensemble Autoencoder

# EnsembleAutoencoder combines multiple architectures for robustness
ensembleAutoencoder = EnsembleAutoencoder(
    n_autoencoders=3,
    hidden_dims=[[24, 12, 8, 4], [64, 32, 16, 4], [20, 10, 5]],
    encoder_activations=['relu', 'tanh', 'relu'],
    decoder_activations=['relu', 'tanh', 'relu'],
    optimizers=['adam', 'sgd', 'rmsprop'],
    learning_rates=[0.001, 0.01, 0.005],
    epochs_list=[10, 15, 20],
    val_size_list=[0.15, 0.15, 0.15],
    stopping_patients=[30, 30, 15],
    dropout_rates=[0.1, 0.2, 0.3],
    batch_sizes=[None, None, 128],
    use_batch_norm=[False, False, False]
)

# Train the ensemble
ensembleAutoencoder.fit(X_train)
print("Ensemble autoencoder trained successfully!")

############################################################################################
### AutoEncoder Model: Custom Basic Model

# Define encoder architecture
encoder = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(24, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu')
])

# Define decoder architecture
decoder = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dense(24, activation='relu'),
    Dense(32, activation='relu'),
    Dense(X_train.shape[1], activation='sigmoid')
])

# Combine into autoencoder
basicAutoencoder = Sequential([encoder, decoder])
basicAutoencoder.compile(optimizer='adam', loss='mse')

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
    performance_metric='Accuracy'
)

# Using Enums as input parameters (recommended for new code)
# optimizer = SegmentAE_Optimizer(
#     autoencoder_models=[ensembleAutoencoder, basicAutoencoder],
#     n_clusters_list=[1, 2, 3],
#     cluster_models=[ClusterModel.KMEANS, ClusterModel.MINIBATCH_KMEANS, ClusterModel.GMM],
#     threshold_ratios=[1, 1.5, 2, 4],
#     performance_metric=PerformanceMetric.ACCURACY
# )

# Run optimization
print("\nRunning grid search optimization...")
sg = optimizer.optimize(X_train, X_test, y_test)

# Access optimization results
preds_test = sg.preds_test
recon_metrics_test = sg.reconstruction_test
leaderboard = optimizer.leaderboard

print("\n" + "="*50)
print("OPTIMIZATION RESULTS")
print("="*50)
print(f"\nBest Performance: {optimizer.best_performance:.4f}")
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

predictions = sg.detections(
    input_data=X_test,
    threshold_ratio=threshold
)

# Calculate final metrics
final_metrics = metrics_classification(y_test, predictions['Predicted Anomalies'])
print("\nFinal Test Performance:")
for metric, value in final_metrics.items():
    print(f"  {metric}: {value:.4f}")























































