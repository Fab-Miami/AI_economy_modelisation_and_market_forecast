import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from model_configuration import Informer
from prepare_data import create_dataset
from tools.tool_fct import *
import matplotlib.pyplot as plt
from tqdm import tqdm


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
INPUT_MONTHS = 36
OUTPUT_MONTHS = 6
BATCH_SIZE = 32 # the model processes BATCH_SIZE different sequences of INPUT_MONTHS months each at a time.
LEARNING_RATE = 0.001
PATIENCE = 50 # how many epochs the validation loss should not improve before the training stops
SCHEDULER_PATIENCE = 20 # how many epochs the validation loss should not improve before the learning rate is reduced
#
EMBEDING_SIZE = 2048 # size of the embeddings
ATTENTION_HEADS = 16 # number of attention heads
LAYER_COUNT = 6 # number of layers
DROPOUT_RATE = 0.2
#
DEFAULTS_EPOCHS = 1000
DEFAULTS_PERCENTAGE_DATA_USED_FOR_TRAINING = .8
#
parameters = parse_parameters(sys.argv[1:]) # param passed from command line
print("Parsed parameters:", parameters)
QUESTIONS = False if 'questions' in parameters and parameters['questions'] else True
EPOCHS = int(parameters.get('epochs') or parameters.get('epoch') or DEFAULTS_EPOCHS)
PERCENTAGE_DATA_USED_FOR_TRAINING = float(parameters.get('percentage_training') or DEFAULTS_PERCENTAGE_DATA_USED_FOR_TRAINING)
# --------------------------------------------------------------------------------------------------------

# Prepare the dataset or get it from file
console.print("Generate dataset_training (g) or download it (d):", style="bold yellow")
mode = input().lower()
if mode == 'g':
    dataset_training = create_dataset(QUESTIONS)
else:
    dataset_training = pd.read_csv('dataset_training.csv', index_col=0, parse_dates=True)

data_tensor = torch.tensor(dataset_training.values, dtype=torch.float32)

def create_sequences(data, input_length, output_length):
    xs, ys = [], []
    for i in range(len(data) - input_length - output_length + 1):
        x = data[i:i+input_length]
        y = data[i+input_length:i+input_length+output_length]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# Create sequences
X, y = create_sequences(data_tensor, INPUT_MONTHS, OUTPUT_MONTHS)

# Split into training and test sets
train_size = int(len(X) * PERCENTAGE_DATA_USED_FOR_TRAINING)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataset = TensorDataset(X_test, y_test)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Instantiate the Informer model
model = Informer(input_dim=X.shape[-1], output_dim=X.shape[-1], d_model=EMBEDING_SIZE, n_heads=ATTENTION_HEADS, n_layers=LAYER_COUNT, dropout=DROPOUT_RATE, factor=5)

# Loss function
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # weight decay of the optimizer to prevent overfitting
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=0.5)


# Training part starts here
def calculate_metrics(output, target):
    mse = nn.MSELoss()(output, target)
    mae = nn.L1Loss()(output, target)
    rmse = torch.sqrt(mse)
    return mse.item(), mae.item(), rmse.item()

# Warmup function
def warmup_lambda(epoch):
    warmup_epochs = 10
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

# Create warmup scheduler
warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

best_val_loss = float('inf')
no_improve_epochs = 0
train_losses = []
val_mses = []
val_maes = []
val_rmses = []

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for X_batch, y_batch in progress_bar:
        optimizer.zero_grad()
        output = model(X_batch, OUTPUT_MONTHS)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()
        epoch_loss += loss.item()

        warmup_scheduler.step()

    avg_epoch_loss = epoch_loss / len(train_loader)

    model.eval()
    val_mse, val_mae, val_rmse = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch, OUTPUT_MONTHS)
            batch_mse, batch_mae, batch_rmse = calculate_metrics(output, y_batch)
            val_mse += batch_mse
            val_mae += batch_mae
            val_rmse += batch_rmse

    avg_val_mse = val_mse / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)
    avg_val_rmse = val_rmse / len(val_loader)

    # Use MSE for learning rate scheduling
    scheduler.step(avg_val_mse)

    train_losses.append(avg_epoch_loss)
    val_mses.append(avg_val_mse)
    val_maes.append(avg_val_mae)
    val_rmses.append(avg_val_rmse)
    

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_epoch_loss:.4f}, "
          f"Val MSE: {avg_val_mse:.4f}, Val MAE: {avg_val_mae:.4f}, Val RMSE: {avg_val_rmse:.4f}")

    # Save the best model based on validation MSE
    if avg_val_mse < best_val_loss:
        best_val_loss = avg_val_mse
        torch.save(model.state_dict(), 'best_informer_model.pth')
        print(f"New best model saved with validation MSE: {best_val_loss:.4f}")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    # Plot and save training statistics after each epoch
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_mses, label='Validation MSE')
    plt.plot(val_maes, label='Validation MAE')
    plt.plot(val_rmses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training Progress - Epoch {epoch+1}')
    plt.savefig('training_progress.png')
    plt.close()
    
    if no_improve_epochs >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("Training completed. Best model saved as 'best_informer_model.pth'")