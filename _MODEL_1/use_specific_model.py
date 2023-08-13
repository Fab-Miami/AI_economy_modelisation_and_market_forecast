import os
from rich.console import Console
from rich.table import Table
from keras.models import load_model
from keras.optimizers import Adam
from MAIN__ import create_data_set
from tools.lstm_V1 import *

console = Console()

# List all directories in the models directory
model_path = '/Users/c/Desktop/AI/proto1/models'
models = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]

# Print models with numbers
table = Table(show_header=True, header_style="bold magenta")
table.add_column("ID", style="dim", width=10)
table.add_column("Model")

for i, model_dir in enumerate(models, 1):
    table.add_row(f"{i}", f"{model_dir}")

console.print(table)

# Ask the user to choose a model
console.print("Choose a model number to load:", style="bold yellow")
choice = int(input()) - 1

# Check if the choice is valid
if 0 <= choice < len(models):
    model_dir_to_load = os.path.join(model_path, models[choice])
    model = load_model(model_dir_to_load, custom_objects={'Adam': Adam})
    print(f"Model {models[choice]} has been loaded!")
    print("\n\nModel Summary:")
    model.summary()  # Print a summary of the model's architecture
else:
    print("Invalid choice!")

# ------------------------------------------------

# get the data:
data_set, original_max_values, original_min_values, final_train_values = create_data_set()

print("\n\nlen(data_set) = ", len(data_set))

# # Prepare the features (RSI, MACD, etc.)
# features = [col for col in data_set.columns if '-' in col]
# features = sorted(features, key=lambda x: x.split('-')[0])
# # Create 3D array
# X_new = []
# for feature in features:
#     X_new.append(new_data_set[feature].values)
# X_new = np.array(X_new).T.reshape(-1, len(features), 1)

# # Now you can use the model to make predictions on this new data
# predictions = model.predict(X_new)

# # If you want to inverse the scaling and percent change transformation, you can use the same logic as in the use_model function
# # Assuming max_price and min_price are the original maximum and minimum values of SPX_close for the new data
# predictions_orig = predictions * (max_price - min_price) + min_price
# predictions_orig = inverse_pct_change(predictions_orig, final_train_values['market_features']['SPX_close'])

