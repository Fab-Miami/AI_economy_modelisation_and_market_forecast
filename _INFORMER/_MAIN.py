import sys
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prepare_data import create_dataset
from tools.tool_fct import *
from model_configuration import Informer

# *************************************************************************************************
# ONLY works with Python 3.9X, not above
# Create the venv like so: /usr/bin/python3/python3.9 -m venv .venv
# pip install requirements.txt
# source .venv/bin/activate 
#
# cd _LSTM_1
# From this folder: pip freeze > ../requirements.txt
#
# python _MAIN.py questions=1 epochs=100 percentage_training=0.8 sequence_length=12
#
#
# AND TO TEST:
# python use_specific_model.py
#
# *************************************************************************************************


# ------------------------------- PARAMETERS -----------------------------------------------------------
DEFAULTS_EPOCHS = 300
DEFAULTS_PERCENTAGE_DATA_USED_FOR_TRAINING = .8
DEFAULTS_MONTH_SEQUENCE_LENGTH = 24 #120
BATCH_SIZE = 32 # the model processes BATCH_SIZE different sequences of MONTH_SEQUENCE_LENGTH months each at a time.
#
parameters = parse_parameters(sys.argv[1:]) # param passed from command line
print("Parsed parameters:", parameters)
QUESTIONS = False if 'questions' in parameters and parameters['questions'] else True
EPOCHS = int(parameters.get('epochs') or parameters.get('epoch') or DEFAULTS_EPOCHS)
PERCENTAGE_DATA_USED_FOR_TRAINING = float(parameters.get('percentage_training') or DEFAULTS_PERCENTAGE_DATA_USED_FOR_TRAINING)
MONTH_SEQUENCE_LENGTH = int(parameters.get('sequence_length') or DEFAULTS_MONTH_SEQUENCE_LENGTH)
# --------------------------------------------------------------------------------------------------------

# Prepare the dataset or get it from file
console.print("Generate dataset_training (g) or download it (d):", style="bold yellow")
mode = input().lower()
if mode == 'g':
    dataset_training = create_dataset(QUESTIONS)
else:
    dataset_training = pd.read_csv('dataset_training.csv', index_col=0, parse_dates=True)

# Convert the dataframe to PyTorch tensors
data_tensor = torch.tensor(dataset_training.values, dtype=torch.float32)

# Prepare data as sequences
# X months are used to predict the next month
# Month X+1 is used as the target value
# The process is repeated for all months in the dataset
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# Setup sequence length and create sequences
X, y = create_sequences(data_tensor, MONTH_SEQUENCE_LENGTH)

# X is ALL the sequences of data points that the model will use to make predictions
# y is ALL the target values that the model will learn to predict


# Split into training and test sets
train_size = int(len(X) * PERCENTAGE_DATA_USED_FOR_TRAINING)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# ---
# print(dataset_training.iloc[:5, 1:])
# print("dataset_training shape:", dataset_training.shape)
# print(data_tensor[:5, :])
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("data_tensor.shape[1]", data_tensor.shape[1])
# print("\n\nFirst sequences in X:", X[:3])  # Print the first 3 sequences
# print("First targets in y:", y[:3])  # Print the first 3 target values
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# ---

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate the Informer model
output_dim = data_tensor.shape[1]
# model = Informer(input_dim=X_train.shape[-1], output_dim=output_dim)
model = Informer(input_dim=X_train.shape[-1], output_dim=output_dim, d_model=1024, n_heads=16, n_layers=3, dropout=0.1, factor=5)

# Define loss function and optimizer.
# GPT: you're using Mean Squared Error (MSE) as the loss function for training your model,
# it might be beneficial to also use other evaluation metrics, such as Mean Absolute Error (MAE)
# or Root Mean Squared Error (RMSE), when you evaluate the performance of your model
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'informer_model.pth')
print("Model saved as 'informer_model.pth'")