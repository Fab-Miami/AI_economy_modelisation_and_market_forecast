import torch
import torch.nn as nn
import torch.nn.functional as F

class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, n_heads=8, n_layers=3, dropout=0.1):
        super(Informer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Embedding layer for the input
        self.embedding = nn.Linear(input_dim, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Fully connected output layer
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, d_model)
        x = x.permute(1, 0, 2)  # Shape: (sequence_length, batch_size, d_model)
        x = self.transformer_encoder(x)  # Shape: (sequence_length, batch_size, d_model)
        x = x[-1, :, :]  # Take the last time step's output (many-to-one)
        x = self.fc(x)  # Shape: (batch_size, output_dim)
        return x
