# conda activate py38

# Gross Domestic Product (GDP): This is a measure of the total economic output of a country. An increase in GDP can be a positive sign for the stock market, as it suggests that the economy is growing.
# Unemployment rate: This is the percentage of the labor force that is actively seeking employment but unable to find it. A lower unemployment rate can be a positive sign for the stock market, as it suggests that the economy is strong and people are able to find jobs.
# Inflation rate: This is the rate at which the general level of prices for goods and services is rising, and subsequently, purchasing power is falling. Central banks attempt to limit inflation, and a low inflation rate is generally seen as a positive sign for the stock market.
# Interest rates: This is the rate at which banks lend money to one another. Changes in interest rates can affect the stock market, as they can influence the cost of borrowing money and the return on investment.
# Corporate earnings: This is the profit that a company generates. Strong corporate earnings can be a positive sign for the stock market, as they suggest that companies are performing well and can potentially pay dividends to shareholders.
# Political events: Political events, such as elections or changes in government policies, can also influence the stock market.
# Consumer spending: Consumer spending is a key driver of economic growth, as it accounts for a large portion of GDP. Factors that can affect consumer spending include income, confidence in the economy, and access to credit.
# Business investment: Business investment, which includes spending on things like machinery, equipment, and buildings, can also be a key driver of economic growth. Factors that can affect business investment include expectations about future demand, access to capital, and the cost of borrowing.
# International trade: The United States is a major player in the global economy and international trade can have a significant impact on the domestic economy. Factors that can affect international trade include exchange rates, tariffs, and economic conditions in other countries.
# debt
# trade balances 
# bonds rates
# Consumer Confidence Index (CCI)
# Corporate earnings
# investor sentiment and risk appetite

# ==> Long Short-Term Memory (LSTM) network (which is a type of RNN). They are able to "remember" important information from earlier in the sequence, which can be useful for predicting future trends based on past data.
# ==> Convolutional neural networks (CNN) which are particularly well-suited for analyzing data with a spatial structure, and autoencoders, which can be used to identify patterns in high-dimensional data.

# improve by adding more granularity to the data and incorporating other data sources.
# instead of using the overall GDP, you could use the GDP growth rate or the GDP per capita, which would give you more information about how the economy is performing.
# credit card transactions
# Technical indicators such as moving averages, relative strength index (RSI), and stochastic oscillator can be used to identify trends and patterns in stock prices and can help to generate buy and sell signals.

# Political events: Political events such as elections, policy changes, and international relations can also have a significant impact on the stock market. You can include data on political events in your model
# Commodity prices: Commodity prices such as oil, gold, and others can be used to make predictions about the stock market as they are closely related to the economy.
# Market volatility: Market volatility can provide insights into investor sentiment and can be used as a feature in your model to predict stock prices.

# LSTM models are not the only option for stock market prediction, other models such as Random Forest, GBM, XGBoost etc. can also be used to generate buy and sell signals. Also, a combination of models such as LSTM with Random Forest can also be used to improve the performance.

#
#  conda activate py38
#
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import aiohttp
import numpy as np
import time
from datetime import datetime
import yfinance as yf
import pandas as pd
from talib import RSI, MACD, BBANDS # technical analysis library
from tools.tool_fct import *
from tools.lstm_V1 import *
from tools.lstm_V2 import *
from tools.transformations import *
from functools import reduce
from dateutil.parser import parse as parse_date
from sklearn.preprocessing import MinMaxScaler
#
from rich import print
from rich.console import Console
console = Console()
#

# *************************************************************************************************
#                                DIFFERENT SOURCES OF SERIES
# -------------------------------------------------------------------------------------------------
fred_series = [
    {'series_name': 'GDP', 'series_id': 'GDPC1', 'frequency': 'q'}, # Gross Domestic Product
    {'series_name': 'GDP per capita', 'series_id': 'A939RX0Q048SBEA', 'frequency': 'q'}, # GDP per capita

    {'series_name': 'Unemployment_Rate', 'series_id': 'UNRATE', 'frequency': 'm'},
    {'series_name': 'Consumer_Price_Index', 'series_id': 'CPIAUCSL', 'frequency': 'm'}, # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
    {'series_name': 'Effective_Rate', 'series_id': 'DFF', 'frequency': 'm'}, # Effective Federal Funds Rate (effective rate)(actual rate that banks lend and borrow from each other)
    # [SAME AS DFF] {'series_name': 'Targeted Rate', 'series_id': 'FEDFUNDS', 'frequency': 'm'}, # Federal Funds Target Rate (rate set by the Federal Reserve) 
    {'series_name': 'Corporate_Profits', 'series_id': 'CP', 'frequency': 'q'},
    # {'series_name': 'NASDAQ', 'series_id': 'NASDAQCOM', 'frequency': 'm'}, # better to get it from Yahoo Finance
    # [NO DATA BEFORE 1986, and almost like NASDAQ anyway] {'series_name': 'NASDAQ100', 'series_id': 'NASDAQ100', 'frequency': 'm'},
    {'series_name': 'Consumer_Confidence_Index', 'series_id': 'CSCICP03USM665S', 'frequency': 'm'}, # Consumer Confidence Index
    {'series_name': 'US_Market_Cap', 'series_id': 'SPASTT01USM661N', 'frequency': 'm'}, # Total Share Prices for All Shares for the United States
    {'series_name': 'Population', 'series_id': 'POP', 'frequency': 'm'}, # US population
    {'series_name': 'US_Debt', 'series_id': 'GFDEGDQ188S', 'frequency': 'q'}, # US Debt
    {'series_name': 'US_Trade_Balance', 'series_id': 'TB3MS', 'frequency': 'm'}, # US Trade Balance
    {'series_name': 'US_Bonds_Rate_10y', 'series_id': 'DGS10', 'frequency': 'm'}, # US Bonds Rate 10 years
    {'series_name': 'US_Bonds_Rate_1y', 'series_id': 'DGS1', 'frequency': 'm'}, # US Bonds Rate 1 year
    {'series_name': 'AAA_Bond_Rate', 'series_id': 'AAA', 'frequency': 'm'}, # AAA Average Corporate Bond Yield
    {'series_name': 'BAA_Bond_Rate', 'series_id': 'BAA', 'frequency': 'm'}, # BAA Average Corporate Bond Yield
    {'series_name': 'Money_Velocity', 'series_id': 'M1V', 'frequency': 'q'}, # Money Velocity (of spending)
    {'series_name': 'Credit_Card_Transactions', 'series_id': 'CCSA', 'frequency': 'm'}, # Credit Card Transactions
    {'series_name': 'PMI_Manufacturing', 'series_id': 'MANEMP', 'frequency': 'm'}, # Manufacturing Employment
    {'series_name': 'Market_Stress', 'series_id': 'STLFSI4', 'frequency': 'm'}, # Stress in the U.S. financial system using a variety of market and economic indicators.
]

quandl_series = [
    {'series_name': 'DOW_JONES', 'series_id': 'DJIA', 'frequency': 'm'}, # DOW JONES
]

# *************************************************************************************************
#                               PARAMETERS and other stuffs
# -------------------------------------------------------------------------------------------------

years_of_history = 50 # back from present

timer_start = time.perf_counter()
loop = asyncio.get_event_loop()

end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
start_date = pd.to_datetime('today') - pd.DateOffset(years=years_of_history)
start_date = start_date.strftime('%Y-%m-%d')

# *************************************************************************************************
#                                     NASDAQ.COM (quandl.com)
# -------------------------------------------------------------------------------------------------
class NasdaqCom:
    def __init__(self, series, observation_start, observation_end):
        api_key = "1PeNtzRGtmacF-LgsG6S"
        self.api_endpoint = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&"
        self.params = {
            "apikey": api_key,                # defined above
            "outputsize": "full",             # get the full-size output
            "datatype": "json",               # return the data in JSON format
            "start_date": observation_start,  # start date for the data
            "end_date": observation_end,      # end date for the data
        }
        self.series = series


# *************************************************************************************************
#                                           FRED
# -------------------------------------------------------------------------------------------------

def get_fred_data():
    file_path = f"../saved_data_from_api/fred_results.csv"
    if os.path.isfile(file_path):
        df_fred = pd.read_csv(file_path, index_col=0)
        df_fred.index = pd.to_datetime(df_fred.index)
        print("[bold yellow]\n============> USING FRED SAVED DATA <============\n[/bold yellow]")
    else:
        fred = FredOnline(fred_series, start_date, end_date)
        df_fred = loop.run_until_complete(fred.get_api_results())
        df_fred.to_csv(file_path)
    return df_fred

class FredOnline:
    def __init__(self, series, observation_start, observation_end):
        api_key = "9e28d63eab23f1dea77320c11110fa4b"
        self.api_endpoint = "https://api.stlouisfed.org/fred/series/observations"
        self.params = {
            "api_key": api_key,                      # defined above
            "observation_start": observation_start,  # start date for the data
            "observation_end": observation_end,      # end date for the data
            "units": "lin",                          # scale the data as a linear series
            "file_type": "json",                     # return the data in JSON format
            "sort_order": "asc",                     # sort the data in ascending order
        }
        self.series = series

    # get results from FRED API, put the results in a dataframe and return it
    async def get_one_series(self, series_name, series_id, frequency='m'):
        async with aiohttp.ClientSession() as session:
            self.params['series_id'] = series_id
            self.params['frequency'] = frequency
            async with session.get(self.api_endpoint, params=self.params) as response:
                try:
                    data = await response.json()
                    df = pd.DataFrame.from_dict(data['observations'])
                    df.set_index('date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                    df['value'] = pd.to_numeric(df['value'], errors='coerce') # convert column value to float, Nan if not possible
                    df.drop(columns=['realtime_start'], inplace=True)
                    df.drop(columns=['realtime_end'], inplace=True)
                    df.rename(columns={'value': series_name}, inplace=True)
                    # fillup missing values if frequency is quarterly
                    if frequency == 'q':
                        date_range = pd.date_range(start=self.params['observation_start'], end=self.params['observation_end'], freq='MS')
                        df = df.reindex(date_range)
                        df = df.ffill() # forward-fill
                    return df
                except Exception as e:
                    print(f"Serie ID: {series_id}, Status: {response.status}, Content type: {response.content_type}")
                    print(await response.text())
                    print(f"Error: {e}")
                    # stop script if there is an error
                    exit()

    # run all the api calls in parallel (async)
    async def get_api_results(self):
        tasks = [self.get_one_series(**one_series) for one_series in self.series]
        # results is an empty dataframe with the same time range defined in the params
        date_range = pd.date_range(start=self.params['observation_start'], end=self.params['observation_end'], freq='MS')
        df_data = pd.DataFrame(index=date_range)
        for task in asyncio.as_completed(tasks): # keeps the order of the results
            df_result  = await task
            df_data = pd.concat([df_data, df_result], axis=1)
        # as some recent values may not be yet known, we fill the last missing values with the last known value
        df_data.iloc[-10:] = df_data.iloc[-10:].fillna(method='ffill')

        # Remove the time from the DateTime index
        df_data.index = pd.to_datetime(df_data.index)
        df_data.index = df_data.index.date
        df_data.index.name = 'Date'

        return df_data


# *************************************************************************************************
#                                       YAHOO FINANCE
# -------------------------------------------------------------------------------------------------

def get_yahoo_data():
    file_path='../saved_data_from_api/yahoo_results.csv'
    if os.path.isfile(file_path):
        df_yahoo = pd.read_csv(file_path, index_col=0)
        df_yahoo.index = pd.to_datetime(df_yahoo.index)
        print("============> USING YAHOO SAVED DATA")
    else:
        df_yahoo = get_online_yahoo_data(start_date)
        df_yahoo.to_csv(file_path)
    return df_yahoo

def get_online_yahoo_data(start_date):
    # get data from yahoo
    sp500  = yf.download('^GSPC', start=start_date)  # daily data

    # resample to monthly
    sp500_monthly = sp500.resample('M').mean()

    # rename columns
    sp500_monthly.rename(columns={'Close': 'SP500_close'}, inplace=True)
    sp500_monthly.rename(columns={'Volume': 'SP500_volume'}, inplace=True)

    # build dataframe with all the data
    df_yahoo = pd.concat([sp500_monthly['SP500_close'],  sp500_monthly['SP500_volume']], axis=1)
    df_yahoo = df_yahoo.tz_localize(None)

    # shift index by one day (mean of December is the value of 1st of January)
    df_yahoo.index = df_yahoo.index + pd.DateOffset(days=1)
    df_yahoo.index.name = 'Date'
    df_yahoo.index = pd.to_datetime(df_yahoo.index)

    return df_yahoo

# *************************************************************************************************
#                                       ELECTION DATA
# -------------------------------------------------------------------------------------------------
def get_elections_data():
    # US elections results (DEM = 1 ; REP = 2)
    df_elections = pd.read_csv('../saved_data_elections/USelections.csv')
    df_elections['Date'] = pd.to_datetime(df_elections['Date'], format='%Y-%m-%d')
    df_elections.index = pd.to_datetime(df_elections["Date"])
    df_elections = df_elections.drop("Date", axis=1)

    df_elections_encoded = one_hot_encode_elections(df_elections)
    df_elections_encoded = df_elections_encoded.astype(int) # convert to int

    return df_elections_encoded

def one_hot_encode_elections(df):
    # One-hot encode the categorical columns
    df_encoded = pd.get_dummies(df, columns=['House Majority', 'Senate Majority', 'Presidency'])
    
    # Rename the columns for better readability
    column_mapping = {
        'House Majority_1': 'House_DEM',
        'House Majority_2': 'House_REP',
        'Senate Majority_1': 'Senate_DEM',
        'Senate Majority_2': 'Senate_REP',
        'Presidency_1': 'President_DEM',
        'Presidency_2': 'President_REP'
    }
    
    df_encoded.rename(columns=column_mapping, inplace=True)
    return df_encoded


# *************************************************************************************************
#                                       GENERATOR DATA
# -------------------------------------------------------------------------------------------------

def get_generator_data():
    file_path = '../saved_data_from_generators/'
    all_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]

    df_list = []

    for file in all_files:
        df = pd.read_csv(os.path.join(file_path, file))
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.index = pd.to_datetime(df["Date"])
        df = df.drop("Date", axis=1)
        df_list.append(df)

    merged_df = pd.concat(df_list, axis=1, join='outer')
    return merged_df


# *************************************************************************************************
#                                       GET DATA FROM STATIC FILES
# -------------------------------------------------------------------------------------------------

def get_static_data():
    file_path = '../saved_data_from_static/'
    all_files = [f for f in os.listdir(file_path) if f.endswith('.csv') and not f.startswith('RAW_')]

    df_list = []

    for file in all_files:
        df = pd.read_csv(os.path.join(file_path, file), parse_dates=['Date'])
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_convert(None)  # convert to naive datetime
        df['Date'] = df['Date'].dt.normalize()  # normalize to midnight (remove time component)
        df.set_index('Date', inplace=True)
        df_list.append(df)

    # merge all dataframes in df_list
    merged_df = pd.concat(df_list, axis=1, join='inner')

    return merged_df


# *************************************************************************************************
# *************************************************************************************************
#                                     - STARTS HERE -
# *************************************************************************************************
# *************************************************************************************************

def create_data_set():

    df_list = ["fred", "elections", "generator", "static"] # "yahoo" is not used anymore as I'm getting SPX from TadingView as a Static download
    df_list = ["static"]
    dfs = {}

    for name in df_list:
        func_name    = f"get_{name}_data"
        df_name      = f"df_{name}"
        # Create a dictionary with the name of the dataframe as key and the dataframe as value
        dfs[df_name] = getattr(sys.modules[__name__], func_name)() # Call the function with the name from df_list ( eg: get_fred_data() )

    # print the dataframes
    for df_name, df in dfs.items():
        print(f"[bold green]\n------------- {df_name.upper()} -------------[/bold green]")
        print(df)
        find_missing_dates(df)

    # plot
    ask_to_plot("Do you want to plot all the data? (yes/no):", dfs)

    # printing missing values
    for name, df in dfs.items():
            print(f"\n[bold red]Missing values in {name}:[/bold red]")
            print(df.isna().sum())

    # merge the dataframes
    print(f"\n[bold yellow]============> MERGING DATAFRAMES <============[bold yellow]")
    data_set = list(dfs.values())[0] # Start with the first dataframe
    data_set.index = pd.to_datetime(data_set.index) # Convert the index to datetime
    data_set.index = data_set.index.to_period('M')  # Convert the index to year-month
    for df in list(dfs.values())[1:]:  # Merge all other dataframes
        df.index = df.index.to_period('M')  # Convert the index to year-month
        data_set = data_set.merge(df, left_index=True, right_index=True, how='inner')
    # convert index back to datetime format with first day of the month
    data_set.index = data_set.index.to_timestamp()

    # find missing dates
    find_missing_dates(data_set)

    # plot
    ask_to_plot("Do you want to plot of the MERGED dataframe? (yes/no):", {'data_set': data_set})

    # add indicators to the dataframe
    data_set = add_indicators(data_set)

    # plot
    ask_to_plot("Do you want to plot of the MERGED dataframe WITH INDICATORS? (yes/no):", {'data_set': data_set})

    # sort columns by name 
    data_set.sort_index(axis=1, inplace=True)

    # backfill missing values
    data_set.fillna(method='bfill', inplace=True)

    # fill NaNs with interpolated values
    print("\n\nNans in data_set before interpolation = ", data_set.isnull().sum().sum())
    data_set.interpolate(method='linear', inplace=True)
    print("Number of Nans after interpolation = ", data_set.isnull().sum().sum(), "\n\n")

    # apply Transformations
    data_set, initial_values = transform_features(data_set)

    # plot
    ask_to_plot("Do you want to plot of the MERGED dataframe WITH TRANSFORMED FEATURES? (yes/no):", {'data_set': data_set}, normalize=False)
    
    # normalize the dataframes
    print(f"[bold yellow]============> NORMALIZING DATAFRAMES <============[bold yellow]\n\n")
    original_max_values = data_set.max()
    original_min_values = data_set.min()
    scaler = MinMaxScaler()
    data_set = pd.DataFrame(scaler.fit_transform(data_set), columns=data_set.columns, index=data_set.index)
    print(f"[bold green]Original max values:\n\n[/bold green]", original_max_values)
    print(f"\n[bold green]Original min values:\n\n[/bold green]", original_min_values)

    # plot
    ask_to_plot("Do you want to plot the FINAL (TRANSFORMED & NORMALIZED) data_set?:", {'data_set': data_set}, normalize=False) # should bbe normalized anyway

    # print time elapsed
    print(f"[blue]Total time elapsed: {time.perf_counter() - timer_start:.2f} seconds\n\n[/blue]")

    return data_set, original_max_values, original_min_values, initial_values


if __name__ == "__main__":
    data_set, original_max_values, original_min_values, initial_values = create_data_set()

    # ------------------------- OUTPUT -----------------------
    console.print("Do you want to trace the Autocorrelation? (1), or Print Info (2), or Nothing (n):", style="bold yellow")
    user_input = input().lower()

    if user_input.lower() == '1':
        autocorrelation(data_set)

    if user_input.lower() == '2':
        print("data_set = ", data_set)
        print("\ndata_set.index", data_set.index)
        print("\ndata_set.columns", data_set.columns)
        print("\nNumber of columns = ", len(data_set.columns))
        print("[bold]---------------------------------------------------------------------------------------------\n\n[/bold]")

    if user_input.lower() == 'n':
        pass

    # --------------------- CREATE THE MODEL -----------------------
    console.print("Do you want to CREATE THE MODEL? (yes/no):", style="bold yellow")
    plot_choice = input().lower()
    if plot_choice == 'y' or plot_choice == 'yes':
        model, X_test, y_test, dates_test = create_the_model_V1(data_set, 50) # dat_set, epochs
        # model, X_test, y_test, dates_test = create_the_model_V2(data_set, 50) # dat_set, epochs
        # current_date with hours, minutes
        model.save(f"../models/model_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.h5")
        print("\n\ndates_test = ", dates_test)
    else:
        console.print("You chose not to run the model. Goodbye.", style="bold cyan")
        sys.exit(0)

    # -------------------- TEST THE MODEL  -----------------------
    max_price = original_max_values['SPX_close']
    min_price = original_min_values['SPX_close']
    test_the_model_V1(model, X_test, y_test, dates_test, max_price, min_price, initial_values)
    # test_the_model_V2(model, X_test, y_test, dates_test, max_price, min_price)
