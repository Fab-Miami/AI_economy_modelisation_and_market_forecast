import os
import json
from art import *
PATH = os.getcwd()
from rich.console import Console
from rich.table import Table
from keras.models import load_model
from keras.optimizers import Adam
from MAIN__ import create_data_set
import matplotlib.pyplot as plt
from tools.lstm_V1 import *
import matplotlib.ticker as mticker
from matplotlib.dates import MonthLocator
console = Console()

#
#  python use_specific_model.py
#

# List all directories in the models directory
model_path = '/Users/c/Desktop/AI/proto1/models'
models = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]

print(f"[bold yellow]\n{text2art('Choose a model', font='small')}[/bold yellow]")
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
    metadata = json.load(open(os.path.join(model_dir_to_load+"/assets/", 'metadata.json'), 'r'))
    last_training_date = pd.Timestamp(metadata['last_training_date'])

    model.summary()  # Print a summary of the model's architecture
else:
    print("Invalid choice!")


# ---------- get the ACTUAL spx ----------
file_path = f'{PATH}/../saved_data_from_static/'
file = "SP500_SPX_1m.csv"
spx_actual_df = pd.read_csv(os.path.join(file_path, file), parse_dates=['Date'])
spx_actual_df['Date'] = pd.to_datetime(spx_actual_df['Date']).dt.tz_convert(None)  # convert to naive datetime
spx_actual_df['Date'] = spx_actual_df['Date'].apply(lambda date: date.replace(day=1))  # set day to 1st of the month
spx_actual_df['Date'] = spx_actual_df['Date'].dt.normalize()  # normalize to midnight (remove time component)
spx_actual_df.set_index('Date', inplace=True)


# ---------- recreate the dataset ----------
# get the data: we are getting the full dataframe, until last month. The goal being to predict the next month
data_set, original_max_values, original_min_values, final_train_values, shift_months = create_data_set()

# prepare the features (RSI, MACD, etc.)
features = [col for col in data_set.columns if '-' in col]
features = sorted(features, key=lambda x: x.split('-')[0])

# create 3D array
X = []
for feature in features:
    X.append(data_set[feature].values)
X = np.array(X).T.reshape(-1, len(features), 1)


# ---------- generate predictions using the model ----------
predictions = model.predict(X)

# inverse the normalization of the predicted values
max_price = original_max_values['SPX_close']
min_price = original_min_values['SPX_close']
predictions_rescaled = predictions * (max_price - min_price) + min_price

# inverse the percent change transformation
# last_date = spx_actual_df.index[-1]
# previous_month_date = last_date - pd.DateOffset(months=12)
# initial_value = spx_actual_df.loc[previous_month_date, 'SPX_close']

first_date = spx_actual_df.index[0]
initial_value = spx_actual_df.loc[first_date, 'SPX_close']

print("-------------------------------------------")
print("initial_value", initial_value)
print(spx_actual_df['SPX_close'].iloc[-12])
print("-------------------------------------------")

# initial_value = final_train_values['market_features']['SPX_close']
predictions_inverse_transformation = inverse_pct_change(predictions_rescaled, initial_value)


# ---------- merge the actual SPX with the predictions ----------
dates = data_set.index
spx_actual_df = spx_actual_df[['SPX_close']]
predictions_df = pd.DataFrame(predictions_inverse_transformation, index=dates, columns=['Predictions'])
merged_df = pd.merge(spx_actual_df, predictions_df, left_index=True, right_index=True, how='outer')
merged_df.fillna(method='ffill', inplace=True) # forward fill missing values


# ---------- plot ----------
merged_df = merged_df.iloc[-36:] # plot only the last 36 months

plt.figure(figsize=(14, 7))
plt.plot(merged_df.index, merged_df['SPX_close'], label="Real Values", color="blue")
plt.plot(merged_df.index, merged_df['Predictions']/1.63, label="Predictions", color="red")  ## /!\/!\/!\/!\/!\/!\/!\/!\ 15.7 is the last value of SPX_close in the training set????

locator = MonthLocator() # One tick per month
plt.gca().xaxis.set_major_locator(locator)

ax = plt.gca()  # Get current axis
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')

plt.axvline(x=last_training_date, color='green', linestyle='--', linewidth=3, label='Last Training Date')

ax.yaxis.set_major_locator(mticker.MultipleLocator(100))

plt.xticks(rotation=45, ha='right')
plt.title("Predicted SPX_close Values")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()  # Ensure layout looks good with rotated labels
plt.show()

# TODO: NEED to draw a fat vertical line to show where the prediction starts