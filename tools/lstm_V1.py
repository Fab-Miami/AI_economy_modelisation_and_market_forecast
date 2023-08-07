import pandas as pd
import numpy as np
#
from rich import print
from rich.console import Console
console = Console()
#
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def create_the_model_V1(data_set, epochs, test_momths=36):
    dates = data_set.index

    # Prepare the features (RSI, MACD, etc.)
    features = [col for col in data_set.columns if '-' in col]
    features = sorted(features, key=lambda x: x.split('-')[0])
    # Create 3D array
    X = []
    for feature in features:
        X.append(data_set[feature].values)
    X = np.array(X).T.reshape(-1, len(features), 1)

    # SPX_close is the target column
    y = data_set['SPX_close'].values

    print("\n\nlen(X) = ", len(X))

    # Train/test split
    train_size = int(len(X) - test_momths)  # -XX months for testing
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = dates[:train_size], dates[train_size:]

    print("\n\ndates_train", dates_train)
    print("dates_test", dates_test, "\n\n")

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=epochs, verbose=3)

    # Evaluation
    loss = model.evaluate(X_test, y_test)
    console.print("Test loss:", loss, style="bold cyan")
    
    return model, X_test, y_test, dates_test



# #############################################################################

def inverse_pct_change(pct_changed_series, initial_value):
    """
    Inverts the percent change transformation.
    """
    # Start with the initial value
    inverted_series = [initial_value]
    
    for change in pct_changed_series:
        # Compute the next value based on the previous value and the percent change
        next_value = inverted_series[-1] * (1 + change)
        inverted_series.append(next_value)
    
    # Remove the initial value
    inverted_series = inverted_series[1:]
    return inverted_series


def test_the_model_V1(model, X_test, y_test, dates_test, max_price, min_price, initial_values=0):
    print("--------------------------------------------")
    print("               TESTING THE MODEL")
    print("--------------------------------------------")
    # 1. Generate predictions using the model
    predictions = model.predict(X_test)

    print("\n\n X_test.shape = ", X_test.shape)
    
    # 2. Inverse the scaling of the predicted and actual values
    # Assuming max_price and min_price are the original maximum and minimum values of SPX_close
    y_test_orig = y_test * (max_price - min_price) + min_price
    predictions_orig = predictions * (max_price - min_price) + min_price

    # 3. Inverse the percent change transformation
    y_test_orig = inverse_pct_change(y_test_orig, initial_values['market_features']['SPX_close'])
    predictions_orig = inverse_pct_change(predictions_orig, initial_values['market_features']['SPX_close'])

    
    # 4. Calculate performance metrics
    mae = mean_absolute_error(y_test_orig, predictions_orig)
    mse = mean_squared_error(y_test_orig, predictions_orig)
    rmse = np.sqrt(mse)
    
    console.print(f"Mean Absolute Error: {mae}", style="bold cyan")
    console.print(f"Mean Squared Error: {mse}", style="bold cyan")
    console.print(f"Root Mean Squared Error: {rmse}", style="bold cyan")
    
    # 5. Plot the real versus predicted values

    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, y_test_orig, label="Real Values", color="blue")
    plt.plot(dates_test, predictions_orig, label="Predictions", color="red", linestyle="dashed")

    # Setting x-axis ticks for every month and rotating them
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major locator to month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Display month and year
    for month_tick in ax.get_xticks():
        plt.axvline(x=month_tick, color='lightgrey', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45) 
    plt.title("Real vs Predicted SPX_close Values")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()  # Ensure layout looks good with rotated labels
    plt.show()
    
    return mae, mse, rmse
