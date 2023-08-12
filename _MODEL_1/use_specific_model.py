import os

from rich.markdown import Markdown
from rich.console import Console
from rich.table import Table
console = Console()

from keras.models import load_model
from keras.optimizers import Adam


# List all files in the models directory
model_path = '/Users/c/Desktop/AI/proto1/models'

models = [f for f in os.listdir(model_path) if f.endswith('.keras')]
# Print models with numbers


table = Table(show_header=True, header_style="bold magenta")
table.add_column("ID", style="dim", width=10)
table.add_column("Model")
table.add_column("Size (KB)")

for i, model_file in enumerate(models, 1):
    file_size = os.path.getsize(os.path.join(model_path, model_file)) / 1024  # Size in KB
    table.add_row(f"{i}", f"{model_file}", f"{file_size}")

console.print(table)

# Ask the user to choose a model
console.print("Choose a model number to load:", style="bold yellow")
choice = int(input()) - 1
# choice = int(input("Choose a model number to load: ")) - 1

# Check if the choice is valid
if 0 <= choice < len(models):
    model_file_to_load = os.path.join(model_path, models[choice])


    model = load_model(model_file_to_load, custom_objects={'Adam': Adam})

    # model = load_model(model_file_to_load)
    print(f"Model {models[choice]} has been loaded!")
    # print("Model Summary:")
    # model.summary()  # Print a summary of the model's architecture
else:
    print("Invalid choice!")

# You can now use the loaded model as needed
