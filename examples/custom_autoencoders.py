import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from segmentae import SegmentAE, Preprocessing, Clustering
from segmentae.data_sources import load_dataset

############################################################################################
### Data Loading

train, test, target = load_dataset(
    dataset_selection='htru2_dataset',  # Options: 'default_credit_card', 'htru2_dataset', 'shuttle_148'
    split_ratio=0.75                          
)

test, future_data = train_test_split(test,
                                     train_size=0.75,
                                     random_state=5)

# Resetting Index is Required
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
future_data = future_data.reset_index(drop=True)

X_train = train.drop(columns=[target]).copy()
X_test, y_test = test.drop(columns=[target]).copy(), test[target].astype(int)
X_future_data = future_data.drop(columns=[target]).copy()

############################################################################################
### Preprocessing with Enums
pr = Preprocessing(
    encoder="IFrequencyEncoder",   # "LabelEncoder", "OneHotEncoder" || #EncoderType.IFREQUENCY,   
    scaler="StandardScaler",       # "MinMaxScaler", "RobustScaler"  || #ScalerType.MINMAX,        
    imputer=None                   # Optional Imputation
)

pr.fit(X=X_train)
X_train = pr.transform(X=X_train)
X_test = pr.transform(X=X_test)
X_future_data = pr.transform(X=X_future_data)

############################################################################################
### Example: Custom Autoencoder Model

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(f"Input shape: {X_train.shape[1]}")

# Define the encoder
encoder = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(24, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu')
])

# Define the decoder
decoder = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dense(24, activation='relu'),
    Dense(32, activation='relu'),
    Dense(X_train.shape[1], activation='sigmoid')
])

# Combine encoder and decoder into an autoencoder model
autoencoder = Sequential([encoder, decoder])

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=None, validation_split=0.1)

############################################################################################
### Clustering Implementation 

cl_model = Clustering(
    cluster_model=["KMeans"],  # "KMeans", "MiniBatchKMeans", "GMM", "Agglomerative"
    n_clusters=3
)

cl_model.clustering_fit(X=X_train)

############################################################################################
### Autoencoder + Clustering Integration

sg = SegmentAE(
    ae_model=autoencoder,
    cl_model=cl_model
)

### Train Reconstruction

# Using ThresholdMetric Enum
sg.reconstruction(
    input_data=X_train,
    threshold_metric="mse" # || "mse", "mae", "rmse", "max_error"  
)

### Reconstruction Error Result Performance

results = sg.evaluation(
    input_data=X_test,
    target_col=y_test,
    threshold_ratio=2.0
)

# Access metadata by cluster
preds_test, recon_metrics_test = sg.preds_test, sg.reconstruction_test

print(f"Evaluation Results:\n{results['global metrics']}")

############################################################################################
### Multiple Threshold Ratio Reconstruction Evaluation

threshold_ratios = [0.75, 1, 1.5, 2, 3, 4]

global_results = pd.concat([
    sg.evaluation(
        input_data=X_test,
        target_col=y_test,
        threshold_ratio=thr
    )["global metrics"]
    for thr in threshold_ratios
])

print("\nPerformance across different thresholds:")
print(global_results)

############################################################################################
### Anomaly Detection Predictions

best_ratio = global_results.sort_values(by="Accuracy", ascending=False).iloc[0]["Threshold Ratio"]
print(f"\nBest threshold ratio: {best_ratio}")

predictions = sg.detections(
    input_data=X_future_data,
    threshold_ratio=best_ratio
)

print(f"Detected {predictions['Predicted Anomalies'].sum()} anomalies out of {len(predictions)} samples")

############################################################################################
### End of Custom Autoencoder Example




