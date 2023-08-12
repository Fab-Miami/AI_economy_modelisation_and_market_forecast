import os
from rich.console import Console
from rich.table import Table
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

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

# You can now use the loaded model as needed
