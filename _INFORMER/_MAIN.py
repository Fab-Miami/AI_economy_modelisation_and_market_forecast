import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from model_configuration import Informer
from prepare_data import create_dataset
from tools.tool_fct import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import L1Loss

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using GPU with MPS")
else:
    device = torch.device("cpu")
    print(f"Using CPU")



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
# Model parameters
INPUT_MONTHS = 12
OUTPUT_MONTHS = 1
LAYER_COUNT = 4 # number of layers
EMBEDDING_SIZE = 2048 # ()>4096< is too big) size of the embeddings
ATTENTION_HEADS = 256 # OK number of attention heads (2048 / 32 = 64, -> 64-dimensional subspace ; which is a reasonable number)
#
# Training parameters
BATCH_SIZE = 16 # the model processes BATCH_SIZE different sequences of INPUT_MONTHS months each at a time.
LEARNING_RATE = 0.0001
PATIENCE = 5000 # how many epochs the validation loss should not improve before the training stops
DROPOUT_RATE = 0.3 # TRY 0.3
DEFAULTS_EPOCHS = 10000
DEFAULTS_PERCENTAGE_DATA_USED_FOR_TRAINING = .8
#
FOCUS_COLUMN = 44 # column for secondary test
#
parameters = parse_parameters(sys.argv[1:]) # param passed from command line
print(f'Prompt parameters: {parameters}')
QUESTIONS = False if 'questions' in parameters and parameters['questions'] else True
EPOCHS = int(parameters.get('epochs') or parameters.get('epoch') or DEFAULTS_EPOCHS)
PERCENTAGE_DATA_USED_FOR_TRAINING = float(parameters.get('percentage_training') or DEFAULTS_PERCENTAGE_DATA_USED_FOR_TRAINING)
# --------------------------------------------------------------------------------------------------------
model_filename = f"best_model_inp{INPUT_MONTHS}_out{OUTPUT_MONTHS}_emb{EMBEDDING_SIZE}_heads{ATTENTION_HEADS}_layers{LAYER_COUNT}_batch{BATCH_SIZE}_rate{LEARNING_RATE}_dropout{DROPOUT_RATE}.pth"
progress_filename = f"2019data_inp{INPUT_MONTHS}_out{OUTPUT_MONTHS}_emb{EMBEDDING_SIZE}_heads{ATTENTION_HEADS}_layers{LAYER_COUNT}_batch{BATCH_SIZE}_rate{LEARNING_RATE}_dropout{DROPOUT_RATE}.png"
# --------------------------------------------------------------------------------------------------------

# Prepare the dataset or get it from file
console.print("Generate dataset_training (g) or download it (d):", style="bold yellow")
mode = input().lower()
if mode == 'g':
    dataset_training = create_dataset(QUESTIONS)
else:
    dataset_training = pd.read_csv('dataset/dataset_training.csv', index_col=0, parse_dates=True)

data_tensor = torch.tensor(dataset_training.values, dtype=torch.float32)

def create_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.to(device)

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
model = Informer(
    input_dim=X.shape[-1], 
    output_dim=X.shape[-1], 
    d_model=EMBEDDING_SIZE,  # Pass EMBEDDING_SIZE as d_model
    n_heads=ATTENTION_HEADS, 
    n_layers=LAYER_COUNT, 
    dropout=DROPOUT_RATE, 
    factor=5,
    l1_lambda=1e-5,
    l2_lambda=1e-4
).to(device)

# Loss function
criterion = nn.MSELoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # weight decay of the optimizer to prevent overfitting

# Training part starts here
def calculate_metrics(output, target):
    mse = nn.MSELoss()(output, target)
    mae = nn.L1Loss()(output, target)
    rmse = torch.sqrt(mse)
    focus_mae = nn.L1Loss()(output[:, FOCUS_COLUMN], target[:, FOCUS_COLUMN])
    return mse.item(), mae.item(), rmse.item(), focus_mae.item()


# Define total steps
total_steps = EPOCHS * len(train_loader)

# Create a combined scheduler with warmup
scheduler = OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=total_steps,
    pct_start=0.3,  # 30% of the steps will be warmup
    anneal_strategy='cos',
    cycle_momentum=False,
    div_factor=25.0,  # initial_lr = max_lr/25
    final_div_factor=1e4,  # min_lr = initial_lr/10000
)

best_val_loss = float('inf')
no_improve_epochs = 0
train_losses = []
val_mses = []
val_maes = []
# val_rmses = []
val_focus_maes = []


l1_lambda = 1e-5  # L1 regularization strength
l2_lambda = 1e-4  # L2 regularization strength
l1_loss = L1Loss(reduction='sum')

# Training loop
print("Starting training...")
mask = create_mask(INPUT_MONTHS)
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for X_batch, y_batch in progress_bar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        # For single-step prediction, we only need the next time step
        y_batch = y_batch[:, 0, :]  # Take only the first step of the target
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient Clipping (was 1.0 )
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)

    model.eval()
    val_mse, val_mae, val_rmse, val_focus_mae = 0, 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch[:, 0, :]  # Take only the first step of the target
            
            output = model(X_batch)
            
            batch_mse, batch_mae, batch_rmse, batch_focus_mae = calculate_metrics(output, y_batch)
            val_mse += batch_mse
            val_mae += batch_mae
            val_rmse += batch_rmse
            val_focus_mae += batch_focus_mae

    avg_val_mse = val_mse / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)
    avg_val_rmse = val_rmse / len(val_loader)
    avg_val_focus_mae = val_focus_mae / len(val_loader)

    train_losses.append(avg_epoch_loss)
    val_mses.append(avg_val_mse)
    val_maes.append(avg_val_mae)
    # val_rmses.append(avg_val_rmse)
    val_focus_maes.append(avg_val_focus_mae)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_epoch_loss:.4f}, "
          f"Val MSE: {avg_val_mse:.4f}, Val MAE: {avg_val_mae:.4f}, Focus MAE: {avg_val_focus_mae:.4f}")

    # Save the best model based on validation MSE
    if avg_val_mse < best_val_loss:
        best_val_loss = avg_val_mse
        torch.save(model.cpu().state_dict(), f'models/{model_filename}')
        model.to(device)  # Move it back to GPU after saving
        print(f"New best model saved as '{model_filename}' with validation MSE: {best_val_loss:.4f}")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    # Plot and save training statistics after each epoch
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_maes, label='Validation MAE', color='lightblue')
    plt.plot(val_mses, label='Validation MSE')
    # plt.plot(val_rmses, label='Validation RMSE')
    plt.plot(val_focus_maes, label=f'SPX MAE', color='red', linewidth=1.5)
    plt.ylim(0, 2)
    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.suptitle(f'Training Progress - Epoch {epoch+1} - Best MSE: {best_val_loss:.4f}')
    plt.title(f'input:{INPUT_MONTHS} output:{OUTPUT_MONTHS} embdedding:{EMBEDDING_SIZE} heads:{ATTENTION_HEADS} layers:{LAYER_COUNT} batch:{BATCH_SIZE} rate:{LEARNING_RATE} dropout:{DROPOUT_RATE}', fontsize=10)
    plt.savefig(progress_filename)
    plt.close()
    
    if no_improve_epochs >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print(f"Training completed. Best model saved as '{model_filename}'")