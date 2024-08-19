import torch
from model_configuration import Informer
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Constants
PERCENTAGE_DATA_USED_FOR_TRAINING = 0.8
MONTH_SEQUENCE_LENGTH = 24
PREDICTION_MONTHS = 6

# Load the dataset
dataset_training = pd.read_csv('dataset_training.csv', index_col=0, parse_dates=True)
data_tensor = torch.tensor(dataset_training.values, dtype=torch.float32)

# Load the scaler used during training
scaler = joblib.load('scaler.joblib')

# Create sequences from the entire dataset
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# Create sequences
X, y = create_sequences(data_tensor, MONTH_SEQUENCE_LENGTH)

# Split into training and test sets
train_size = int(len(X) * PERCENTAGE_DATA_USED_FOR_TRAINING)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Load the model
output_dim = data_tensor.shape[1]
model = Informer(input_dim=X_train.shape[-1], output_dim=output_dim, d_model=1024, n_heads=16, n_layers=3, dropout=0.1, factor=5)
model.load_state_dict(torch.load('informer_model.pth'))
model = model.float()
model.eval()

# Function to make predictions
def make_prediction(model, input_data):
    with torch.no_grad():
        prediction = model(input_data)
    return prediction.squeeze().numpy()

# Make predictions for the next PREDICTION_MONTHS months
predictions = []
actual_values = []

for i in range(PREDICTION_MONTHS):
    input_data = X_test[i:i+1]  # Take one sequence at a time
    prediction = make_prediction(model, input_data)
    predictions.append(prediction)
    actual_values.append(y_test[i].numpy())

predictions = np.array(predictions)
actual_values = np.array(actual_values)

# Inverse transform the predictions and actual values
predictions_original = scaler.inverse_transform(predictions)
actual_values_original = scaler.inverse_transform(actual_values)

# Specify the indices of interest (replace with your actual indices)
indices_of_interest = [28, 43, 52]  # Example indices

# Extract values for the indices of interest
predictions_of_interest = predictions_original[:, indices_of_interest]
actual_values_of_interest = actual_values_original[:, indices_of_interest]

# Plotting
plt.figure(figsize=(15, 10))
for i, index in enumerate(indices_of_interest):
    plt.subplot(len(indices_of_interest), 1, i+1)
    plt.plot(range(PREDICTION_MONTHS), predictions_of_interest[:, i], label='Predicted')
    plt.plot(range(PREDICTION_MONTHS), actual_values_of_interest[:, i], label='Actual')
    plt.title(f'Feature {dataset_training.columns[index]}')
    plt.legend()

plt.tight_layout()
plt.savefig('prediction_vs_actual.png')
plt.show()

print("Predictions for indices of interest:")
print(predictions_of_interest)
print("\nActual values for indices of interest:")
print(actual_values_of_interest)

# Print some statistics to verify scaling
print("\nScaling verification:")
print(f"Original data range: [{data_tensor.min().item()}, {data_tensor.max().item()}]")
print(f"Predicted data range: [{predictions.min()}, {predictions.max()}]")
print(f"Inverse transformed predictions range: [{predictions_original.min()}, {predictions_original.max()}]")
print(f"Actual data range: [{actual_values.min()}, {actual_values.max()}]")
print(f"Inverse transformed actual values range: [{actual_values_original.min()}, {actual_values_original.max()}]")

# Print feature names for the indices of interest
for index in indices_of_interest:
    print(f"\nFeature at index {index}: {dataset_training.columns[index]}")