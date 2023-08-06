def create_data_set():

    df_list = ["fred", "elections", "generator", "static"] 
    dfs = {}

    for name in df_list:
        func_name    = f"get_{name}_data"
        df_name      = f"df_{name}"
        dfs[df_name] = getattr(sys.modules[__name__], func_name)() # Call the function with the name from df_list ( eg: get_fred_data() )


    data_set = list(dfs.values())[0] 
    data_set.index = pd.to_datetime(data_set.index) 
    data_set.index = data_set.index.to_period('M') 
    for df in list(dfs.values())[1:]: 
        df.index = df.index.to_period('M') 
        data_set = data_set.merge(df, left_index=True, right_index=True, how='inner')
    data_set.index = data_set.index.to_timestamp()

    data_set = add_indicators(data_set)
    data_set.sort_index(axis=1, inplace=True)
    data_set.fillna(method='bfill', inplace=True)
    data_set.interpolate(method='linear', inplace=True)
    data_set, initial_values = transform_features(data_set)

    original_max_values = data_set.max()
    original_min_values = data_set.min()
    scaler = MinMaxScaler()
    
    data_set = pd.DataFrame(scaler.fit_transform(data_set), columns=data_set.columns, index=data_set.index)
   
    return data_set, original_max_values, original_min_values, initial_values

def calc_pct_change(df):
    return df.pct_change().dropna()


def calc_diff(df):
    return df.diff().dropna()

def transform_features(data_set):
    macroeconomic_features = ['AAA_Bond_Rate', 'BAA_Bond_Rate', 'Consumer_Confidence_Index', 'Consumer_Price_Index', 
                              'Corporate_Profits', 'GDP', 'GDP per capita', 'Money_Velocity', 
                              'PMI_Manufacturing', 'Population', 'US_Debt', 'US_Market_Cap', 
                              'US_Trade_Balance', 'Unemployment_Rate','US_Bonds_Rate_10y', 'US_Bonds_Rate_1y', 'Credit_Card_Transactions', 'Effective_Rate', 'Market_Stress']
    
    technical_features  = ['DJI-BBlower', 'DJI-BBmiddle', 'DJI-BBupper', 'DJI-MACD', 'DJI-MACDhist', 
                          'DJI-MACDsignal', 'DJI-RSI', 'IXIC-BBlower', 'IXIC-BBmiddle', 'IXIC-BBupper', 
                          'IXIC-MACD', 'IXIC-MACDhist', 'IXIC-MACDsignal', 'IXIC-RSI', 'SPX-BBlower', 
                          'SPX-BBmiddle', 'SPX-BBupper', 'SPX-MACD', 'SPX-MACDhist', 'SPX-MACDsignal', 'SPX-RSI']
    
    market_features = ['DJI_close', 'DJI_volume', 'IXIC_close', 'SPX_close', 'SPX_volume', 'IXIC_volume', 'IXIC_volume', 'Composite_VIX']
    
    political_features = ['House_DEM', 'House_REP', 'President_DEM', 'President_REP', 'Senate_DEM', 'Senate_REP']

    initial_values = {
        'macroeconomic_features': data_set[macroeconomic_features].iloc[0],
        'technical_features': data_set[technical_features].iloc[0],
        'market_features': data_set[market_features].iloc[0]
    }

    data_set_macro = calc_pct_change(data_set[macroeconomic_features])

    data_set_technical = calc_pct_change(data_set[technical_features])

    data_set_market = calc_pct_change(data_set[market_features])

    data_set_political = data_set[political_features] 

    data_set_transformed = pd.concat([data_set_macro, data_set_technical, data_set_market, data_set_political], axis=1)

    return data_set_transformed.dropna(), initial_values 

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

def create_the_model_V1(data_set, epochs):
    dates = data_set.index

    features = [col for col in data_set.columns if '-' in col]
    features = sorted(features, key=lambda x: x.split('-')[0])
    X = []
    for feature in features:
        X.append(data_set[feature].values)
    X = np.array(X).T.reshape(-1, len(features), 1)

    y = data_set['SPX_close'].values

    train_size = int(len(X) * 0.8) 
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = dates[:train_size], dates[train_size:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=64, epochs=epochs)
    loss = model.evaluate(X_test, y_test)
    console.print("Test loss:", loss, style="bold cyan")
    
    return model, X_test, y_test, dates_test



def test_the_model_V1(model, X_test, y_test, dates_test, max_price, min_price, initial_values=0):


