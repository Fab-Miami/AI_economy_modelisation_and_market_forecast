import sys
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from prepare_data import create_data_set
from tools.tool_fct import *
from model_configuration import Informer

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return (self.X[idx:idx + self.sequence_length],
                self.y[idx + self.sequence_length])

class Informer(nn.Module):
    def __init__(self, input_dim, seq_len, output_dim, d_model=512, n_heads=8, e_layers=2, d_ff=2048):
        super(Informer, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, d_ff), num_layers=e_layers)
        self.linear = nn.Linear(seq_len * input_dim, output_dim)

    def forward(self, src):
        # src shape: [batch_size, sequence_length, num_features]
        src = src.permute(1, 0, 2)  # [sequence_length, batch_size, num_features]
        output = self.encoder(src)
        output = output.permute(1, 0, 2).reshape(src.shape[1], -1)
        return self.linear(output)


# *************************************************************************************************
#                                     - STARTS HERE -
# *************************************************************************************************
# ONLY works with Python 3.9X, not above
# Create the venv like so: /usr/bin/python3/python3.9 -m venv .venv
# pip install requirements.txt
# source .venv/bin/activate 
#
# cd _LSTM_1
# From this folder: pip freeze > ../requirements.txt
#
# python MAIN__.py model=1 epochs=100 test_months=24
# python MAIN__.py --model 1 --epochs 100 --test_months 24
#
#
# AND TO TEST:
# python use_specific_model.py
#
# *************************************************************************************************
if __name__ == "__main__":

     # ------------------------- PARAMETERS -----------------------
    DEFAULTS_EPOCHS = 300
    DEFAULTS_TEST_MONTHS = 36
    parameters = parse_parameters(sys.argv[1:]) # param passed from command line
    QUESTIONS = False if 'model' in parameters and parameters['model'] else True

    # Prepare the dataset
    data_set_normalized = create_data_set(QUESTIONS)

    # Convert the dataframe to PyTorch tensors
    data_tensor = torch.tensor(data_set_normalized.values, dtype=torch.float32)

    # Prepare your data as sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    sequence_length = 12  # Example sequence length
    X, y = create_sequences(data_tensor, sequence_length)

    # Split into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Instantiate the Informer model
    model = Informer(input_dim=X_train.shape[-1], output_dim=1)  # Adjust parameters as needed

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'informer_model.pth')
    print("Model saved as 'informer_model.pth'")


    # # Define model parameters
    # sequence_length = 12  # Number of months used to predict the next month
    # input_dim = X_scaled.shape[1]  # Number of features
    # output_dim = 1  # We want to predict a single value (SPY)
    # model = Informer(input_dim=input_dim, seq_len=sequence_length, output_dim=output_dim)

    # # Loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Prepare the dataset and dataloader
    # dataset = TimeSeriesDataset(torch.tensor(X_scaled, dtype=torch.float32),
    #                             torch.tensor(y_scaled, dtype=torch.float32),
    #                             sequence_length=sequence_length)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
