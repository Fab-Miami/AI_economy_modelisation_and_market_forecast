import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Assuming the data is stored in a DataFrame named 'df_results'
# and the SPY data is in a column called 'SP500_Close'

# Step 1: Preprocess the data
df = df_results.dropna()

# Step 2: Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Step 3: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train_data)

# Step 4: Prepare the input data in the required format for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back, 0])  # Assuming the SPY data is the first column
    return np.array(X), np.array(Y)

look_back = 10
X_train, y_train = create_dataset(scaled_data, look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], scaled_data.shape[1]))

# Step 5: Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the model and save it
model.fit(X_train, y_train, epochs=50, batch_size=32)
model.save('lstm_model_1.h5')


# Step 7: Test and evaluate the model (continued)
loaded_model = load_model('lstm_model.h5')

# Use the loaded model for predictions
# For example, you can use it on the test data like this:
predicted = loaded_model.predict(X_test)
predicted = model.predict(X_test)
predicted_prices = np.zeros_like(scaled_test_data)
predicted_prices[:, 0] = predicted[:, 0]
predicted_prices = scaler.inverse_transform(predicted_prices)[:, 0]

# Calculate the mean squared error
mse = mean_squared_error(y_test, predicted)
print(f'Mean squared error: {mse}')

# Step 8: Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plt.plot(df.index[train_size + look_back + 1:], test_data['SP500_Close'].iloc[look_back + 1:], label='Actual')
plt.plot(df.index[train_size + look_back + 1:], predicted_prices[look_back + 1:], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
