from segmentae.anomaly_detection import (Preprocessing,
                                         EnsembleAutoencoder,
                                         )
from segmentae.optimizer import SegmentAE_Optimizer
from segmentae.data_sources.examples import load_dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense

############################################################################################
### Data Loading

train, test, target = load_dataset(dataset_selection = 'network_intrusions', 
                                   split_ratio = 0.75) # Options | 'network_intrusions', 'default_credit_card', 
                                                       #         | 'htru2_dataset', 'shuttle_148' 

X_train = train.drop(columns=[target]).copy()
X_test = test.drop(columns=[target]).copy()

#y_train = train[target].astype(int)
y_test = test[target].astype(int)

############################################################################################
### Preprocessing

pr = Preprocessing(encoder = "IFrequencyEncoder", # Options | "IFrequencyEncoder", "LabelEncoder", "OneHotEncoder", None
                   scaler = "MinMaxScaler",       # Options | "MinMaxScaler", "StandardScaler", "RobustScaler", None
                   imputer = None)                # Options | "Simple", "RandomForest", "ExtraTrees", "GBR", "KNN",
                                                  #         | "XGBoost","Lightgbm","Catboost", None

pr.fit(X = X_train)
X_train = pr.transform(X = X_train)
X_test = pr.transform(X = X_test)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#########################################################
### AutoEncoder Model 1

ensembleAutoencoder = EnsembleAutoencoder(
    n_autoencoders = 3,
    hidden_dims = [[24, 12, 8, 4], [64, 32, 16, 4], [20, 10, 5]],
    encoder_activations = ['relu', 'tanh', 'relu'],
    decoder_activations = ['relu', 'tanh', 'relu'],
    optimizers = ['adam', 'sgd', 'rmsprop'],
    learning_rates = [0.001, 0.01, 0.005],
    epochs_list = [10, 15, 20],
    val_size_list = [0.15, 0.15, 0.15],
    stopping_patients = [30, 30, 15],
    dropout_rates = [0.1, 0.2, 0.3],
    batch_sizes = [None, None, 128],
    use_batch_norm = [False, False, False]
)

# Autoencoder Implementation
ensembleAutoencoder.fit(X_train)

#########################################################
### AutoEncoder Model 2 (Basic Built-in Model) 

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
basicAutoencoder = Sequential([encoder, decoder])

# Compile the model
basicAutoencoder.compile(optimizer='adam', loss='mse')

# Train the model
basicAutoencoder.fit(X_train, X_train, epochs=50, batch_size=None, validation_split=0.1)

############################################################################################
### SegmentAE Optimizer Implementation

optimizer = SegmentAE_Optimizer(autoencoder_models = [ensembleAutoencoder, basicAutoencoder],
                                n_clusters_list = [1, 2, 3],
                                cluster_models = ["KMeans", "MiniBatchKMeans", "GMM"],  # Options | KMeans, MiniBatchKMeans, GMM, Agglomerative
                                threshold_ratios = [1, 1.5, 2, 4],
                                performance_metric = 'Accuracy')
sg = optimizer.optimize(X_train, X_test, y_test)

preds_test, recon_metrics_test = sg.preds_test, sg.reconstruction_test # Test Metadata by Cluster
leaderboard = optimizer.leaderboard                                    # Leaderboard Combinations Result
print(leaderboard.head(10).to_string())

############################################################################################
### Anomaly Detection Predictions

threshold = optimizer.best_threshold_ratio

predictions = sg.detections(input_data = X_test,
                            threshold_ratio = threshold)



