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

# *********************************************************************
#
#   SPLIT THE DATA SET IN TRAINING from T0 until now minus XX months
#
# *********************************************************************

def create_the_model_V2(data_set, epochs):
    dates = data_set.index
    # Prepare your features
    features = [col for col in data_set.columns if '-' in col]
    features = sorted(features, key=lambda x: x.split('-')[0])

    print("\n\nfeatures = ", features, "\n\n")

    # Create 3D array
    X = []
    for feature in features:
        X.append(data_set[feature].values)
    X = np.array(X).T.reshape(-1, len(features), 1)

    # SPX_close is the target column
    y = data_set['SPX_close'].values

    # Train/test split
    train_size = len(X) - 36  # All but the last XX months for training
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
    model.fit(X_train, y_train, batch_size=64, epochs=epochs)

    # Evaluation
    loss = model.evaluate(X_test, y_test)
    console.print("Test loss:", loss, style="bold cyan")
    
    return model, X_test, y_test, dates_test



def test_the_model_V2(model, X_test, y_test, dates_test, max_price, min_price):
    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Rescaling the predictions back to the original scale
    y_test_rescaled = y_test * (max_price - min_price) + min_price
    y_pred_rescaled = y_pred * (max_price - min_price) + min_price

    # Calculating MAE and RMSE
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

    console.print(f"MAE: {mae}", style="bold blue")
    console.print(f"RMSE: {rmse}", style="bold  blue")

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(dates_test, y_test_rescaled, color='blue', label='Actual SPX close price')
    plt.plot(dates_test, y_pred_rescaled, color='red', label='Predicted SPX close price')
    plt.title('SPX close price prediction')
    plt.xlabel('Time')
    plt.ylabel('SPX close price')
    plt.grid(True)
    plt.legend()
    plt.show()