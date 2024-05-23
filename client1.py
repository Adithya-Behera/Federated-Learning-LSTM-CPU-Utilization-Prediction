import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
logging.getLogger("flwr").setLevel(logging.WARNING)
# Load data
data = pd.read_csv("data/1.csv")
data = data[["Timestamp [ms]", "CPU usage [%]"]].dropna()  # Select relevant features and remove NaN values

# Ensure there are no infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler()
data[["Timestamp [ms]", "CPU usage [%]"]] = scaler.fit_transform(data[["Timestamp [ms]", "CPU usage [%]"]])

# Prepare data for LSTM
window_size = 10  # Define window size for LSTM
X = []
y = []
for i in range(len(data) - window_size):
    X.append(data.iloc[i:i+window_size, 0].values)  # Timestamp [ms]
    y.append(data.iloc[i+window_size, 1])  # CPU usage [%]
X = np.array(X)
y = np.array(y)

# Reshape input data for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Define split point for train and test sets
split_point = int(len(X) * 0.7)  # 70% train, 30% test

# Split data into train and test sets
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Clip gradients to prevent exploding gradients
model.compile(optimizer=optimizer, loss='mse')

# Define Flower client
class TimeSeriesClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=0)
        hist = r.history
        print("Fit history:", hist)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Evaluation loss: {loss}")
        return loss, len(X_test), {'loss': loss}

# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=TimeSeriesClient(),
    grpc_max_message_length=1024*1024*1024,
)

