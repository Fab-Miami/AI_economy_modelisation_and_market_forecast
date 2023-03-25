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

import asyncio
import aiohttp
import nasdaqdatalink
import time
import os
import yfinance as yf
import pandas as pd
from talib import RSI, MACD, BBANDS # technical analysis library
from model_lstm_1 import lstm_1
from useful_fct import autocorrelation , plot_columns, plot_columns_scaled

# *************************************************************************************************
#                                        SERIES
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
    # [NO DATA BEFORE FEV 2013] {'series_name': 'Dow Jones', 'series_id': 'DJIA', 'frequency': 'm'},
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
    {'series_name': 'Market_Stress', 'series_id': 'STLFSI', 'frequency': 'm'}, # Stress in the U.S. financial system using a variety of market and economic indicators.
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

def get_fred_data(fred_series, start_date, end_date, loop, file_path = 'saved_data_api/fred_results.csv'):
    if os.path.isfile(file_path):
        df_fred = pd.read_csv(file_path, index_col=0)
        # df_results.index = pd.to_datetime(df_results.index)
        print("============ USING FRED SAVED DATA ============")
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
        df_results = pd.DataFrame(index=date_range)
        for task in asyncio.as_completed(tasks): # keeps the order of the results
            df_result  = await task
            df_results = pd.concat([df_results, df_result], axis=1)
        # as some recent values may not be yet known, we fill the last missing values with the last known value
        df_results.iloc[-10:] = df_results.iloc[-10:].fillna(method='ffill')
        df_results.index.name = 'Date'
        df_results.index = pd.to_datetime(df_results.index)

        return df_results

# *************************************************************************************************
#                                       YAHOO FINANCE
# -------------------------------------------------------------------------------------------------

def get_yahoo_data(start_date, file_path='saved_data_api/yahoo_results.csv'):
    if os.path.isfile(file_path):
        df_yahoo = pd.read_csv(file_path, index_col=0)
        print("============ USING YAHOO SAVED DATA ============")
    else:
        df_yahoo = get_online_yahoo_data(start_date)
        df_yahoo.to_csv(file_path)
    return df_yahoo

def get_online_yahoo_data(start_date):
    # get data from yahoo
    dow             = yf.download('^DJI', start=start_date)   # daily data
    sp500           = yf.download('^GSPC', start=start_date)  # daily data
    vix             = yf.download('^VIX', start=start_date)   # daily data
    nasdaq          = yf.download('^IXIC', start=start_date)  # daily data

    # resample to monthly
    dow_monthly     = dow.resample('M').mean()
    sp500_monthly   = sp500.resample('M').mean()
    vix_monthly     = vix.resample('M').mean()
    nasdaq_monthly  = nasdaq.resample('M').mean()

    # rename columns
    sp500_monthly.rename(columns={'Close': 'SP500_Close'}, inplace=True)
    sp500_monthly.rename(columns={'Volume': 'SP500_Volume'}, inplace=True)
    dow_monthly.rename(columns={'Close': 'DOW_Close'}, inplace=True)
    dow_monthly.rename(columns={'Volume': 'DOW_Volume'}, inplace=True)
    nasdaq_monthly.rename(columns={'Close': 'NASDAQ_Close'}, inplace=True)
    nasdaq_monthly.rename(columns={'Volume': 'NASDAQ_Volume'}, inplace=True)
    # VIX is not a stock, so no volume
    vix_monthly.rename(columns={'Close': 'VIX_Close'}, inplace=True)

    # build dataframe with all the data
    df_yahoo = pd.concat([dow_monthly['DOW_Close'], dow_monthly['DOW_Volume'], sp500_monthly['SP500_Close'],  sp500_monthly['SP500_Volume'], nasdaq_monthly['NASDAQ_Close'],  nasdaq_monthly['NASDAQ_Volume'], vix_monthly['VIX_Close']], axis=1)
    df_yahoo = df_yahoo.tz_localize(None)

    # shift index by one day (mean of December is the value of 1st of January)
    df_yahoo.index = df_yahoo.index + pd.DateOffset(days=1)
    df_yahoo.index.name = 'Date'
    df_yahoo.index = pd.to_datetime(df_yahoo.index)

    return df_yahoo

# *************************************************************************************************
#                                       ELECTION DATA
# -------------------------------------------------------------------------------------------------
def get_election_data():
    # US elections results (DEM = 1 ; REP = 2)
    df_elections = pd.read_csv('saved_data_static/USelections.csv')
    df_elections['Date'] = pd.to_datetime(df_elections['Date'], format='%Y-%m-%d')
    df_elections.index = pd.to_datetime(df_elections["Date"])
    df_elections = df_elections.drop("Date", axis=1)
    return df_elections

# *************************************************************************************************
# *************************************************************************************************
#                                     - STARTS HERE -
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

df_fred         = get_fred_data()
df_yahoo        = get_yahoo_data()
df_elections    = get_election_data()

print(df_fred)
print("--------------------------------")
print(df_fred.index.dtype)
print("================================")

print(df_yahoo)
print("--------------------------------")
print(df_yahoo.index.dtype)
print("================================")

print(df_elections)
print("--------------------------------")
print(df_elections.index.dtype)
print("================================")

# merge those dataframes
df_results = pd.concat([df_fred, df_yahoo], axis=1, join='inner')
df_results = pd.concat([df_results, df_elections], axis=1, join='inner')


print("--------------------------------")
print("--------------------------------")
print(df_results)

# stop script execution here
import sys
sys.exit()


# add RSI
df_results['SP500-RSI'] = RSI(df_results['SP500_Close'], timeperiod=14)
df_results['DOW-RSI'] = RSI(df_results['DOW_Close'], timeperiod=14)

# add MACD
df_results['SP500-MACD'], df_results['SP500-MACDsignal'], df_results['SP500-MACDhist'] = MACD(df_results['SP500_Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df_results['DOW-MACD'], df_results['DOW-MACDsignal'], df_results['DOW-MACDhist'] = MACD(df_results['DOW_Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# add Bollinger Bands
df_results['SP500-BBupper'], df_results['SP500-BBlower'], df_results['SP500-BBmiddle'] = BBANDS(df_results['SP500_Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
df_results['DOW-BBupper'], df_results['DOW-BBlower'], df_results['DOW-BBmiddle'] = BBANDS(df_results['DOW_Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# sort columns by name 
df_results.sort_index(axis=1, inplace=True)

# backfill missing values
df_results.fillna(method='bfill', inplace=True)


print(f"Total time elapsed: {time.perf_counter() - timer_start:.2f} seconds")
print(df_results.columns)


# autocorrelation(df_results)
# plot_columns_scaled(df_results, ['Unemployment Rate', 'Targeted Rate', 'Effective Rate'])
plot_columns_scaled(df_results)