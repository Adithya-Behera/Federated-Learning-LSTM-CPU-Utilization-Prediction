# Federated LSTM CPU Utilization Prediction

This project implements a federated learning approach using LSTM (Long Short-Term Memory) neural networks to predict CPU utilization based on time series data. The federated learning framework is implemented using the Flower framework.

## Dependencies

Make sure you have the following dependencies installed:
- Python 3.x
- TensorFlow
- Flower

You can install Flower using pip:
- pip install flwr

## Dataset

The dataset consists of time series data for CPU utilization. Each client has its own dataset, which is a subset of the overall time series data.

## Running the Server

To run the Flower server, execute the following command:
- python server.py
  
The server will listen for incoming connections from clients and coordinate the federated learning process.

## Running the Clients

To run the Flower clients, execute the following command for each client:
- python client1.py
- python client2.py

Each client will connect to the server and participate in the federated learning process using its own dataset.
