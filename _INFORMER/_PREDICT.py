import torch
from model_configuration import Informer
import pandas as pd
import joblib

PERCENTAGE_DATA_USED_FOR_TRAINING = 0.8

# I need a rolling window

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

# Setup sequence length (same as during training) and create sequences
MONTH_SEQUENCE_LENGTH = 24  # Use the same sequence length as in training
X, y = create_sequences(data_tensor, MONTH_SEQUENCE_LENGTH)

# Split into training and test sets
PERCENTAGE_DATA_USED_FOR_TRAINING = 0.8
train_size = int(len(X) * PERCENTAGE_DATA_USED_FOR_TRAINING)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Reshape X_test to 2D for scaling (batch_size * sequence_length, n_features)
X_test_reshaped = X_test.view(-1, X_test.shape[-1])

# Scale the reshaped test set
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)

# Reshape back to 3D (batch_size, sequence_length, n_features)
X_test_scaled = torch.tensor(X_test_scaled_reshaped).view(X_test.shape)

# Ensure the tensor is float32
X_test_scaled = X_test_scaled.float()

# Load the model
output_dim = data_tensor.shape[1] # determine the output dimension
model = Informer(input_dim=X_train.shape[-1], output_dim=output_dim, d_model=1024, n_heads=16, n_layers=3, dropout=0.1, factor=5)

# Load model weights and ensure they are in float32
model.load_state_dict(torch.load('informer_model.pth'))
model = model.float()  # Convert model parameters to float32

model.eval()  # Set the model to evaluation mode

# Take the first 12 months of the test set for prediction
input_data = X_test_scaled[:12]  # Ensure this is of shape (batch_size, sequence_length, n_features)

# Make the prediction
with torch.no_grad():
    prediction_scaled = model(input_data)

# Convert the prediction back to the original scale
prediction_numpy = prediction_scaled.squeeze(0).numpy()
prediction_original_scale = scaler.inverse_transform(prediction_numpy)

print("Prediction for the first 12 months of the test set (original scale):")
print(prediction_original_scale)
