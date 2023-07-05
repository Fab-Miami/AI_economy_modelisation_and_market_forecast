import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def run_the_model(data_set, epochs):
    # Prepare your features
    features = [col for col in data_set.columns if '-' in col]
    features = sorted(features, key=lambda x: x.split('-')[0])

    # Create 3D array
    X = []
    for feature in features:
        X.append(data_set[feature].values)
    X = np.array(X).T.reshape(-1, len(features), 1)

    # SPX_close is the target column
    y = data_set['SPX_close'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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
    print("Test loss:", loss)
    
    return model, X_test, y_test


def test_the_model(model, X_test, y_test, max_price, min_price):
    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Rescaling the predictions back to the original scale
    y_test_rescaled = y_test * (max_price - min_price) + min_price
    y_pred_rescaled = y_pred * (max_price - min_price) + min_price

    # Calculating MAE and RMSE
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, color='blue', label='Actual SPX close price')
    plt.plot(y_pred_rescaled, color='red', label='Predicted SPX close price')
    plt.title('SPX close price prediction')
    plt.xlabel('Time')
    plt.ylabel('SPX close price')
    plt.grid(True)
    plt.legend()
    plt.show()
