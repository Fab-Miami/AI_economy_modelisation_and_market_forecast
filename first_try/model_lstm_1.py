import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense
from keras.models import Sequential

def lstm_1(data):

    data = pd.merge(economic_data, index_data, on='date')
    time_series_data = data.set_index('date').astype(float).diff().dropna()

    # Split the data into training and testing sets
    train_data = time_series_data[:'2018']
    test_data = time_series_data['2019':]