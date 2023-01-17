# conda activate py38

# passphrase: azerty

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


import asyncio
import aiohttp
import time
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI

# *************************************************************************************************
#                                        SERIES
# -------------------------------------------------------------------------------------------------
series = [
    {'series_name': 'GDP', 'series_id': 'GDPC1', 'frequency': 'q'}, # Gross Domestic Product
    {'series_name': 'Unemployment Rate', 'series_id': 'UNRATE', 'frequency': 'm'},
    {'series_name': 'Consumer Price Index', 'series_id': 'CPIAUCSL', 'frequency': 'm'}, # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
    {'series_name': 'Effective Rate', 'series_id': 'DFF', 'frequency': 'm'}, # Effective Federal Funds Rate (effective rate)(actual rate that banks lend and borrow from each other)
    {'series_name': 'Targeted Rate', 'series_id': 'FEDFUNDS', 'frequency': 'm'}, # Federal Funds Target Rate (rate set by the Federal Reserve)
    {'series_name': 'Corporate Profits', 'series_id': 'CP', 'frequency': 'q'},
    {'series_name': 'NASDAQ', 'series_id': 'NASDAQCOM', 'frequency': 'm'},
    {'series_name': 'NASDAQ100', 'series_id': 'NASDAQ100', 'frequency': 'm'},
    {'series_name': 'S&P500', 'series_id': 'SP500', 'frequency': 'm'},
    {'series_name': 'Dow Jones', 'series_id': 'DJIA', 'frequency': 'm'},
    {'series_name': 'Consumer Confidence Index', 'series_id': 'CSCICP03USM665S', 'frequency': 'm'}, # Consumer Confidence Index
    {'series_name': 'US Market Cap', 'series_id': 'SPASTT01USM661N', 'frequency': 'm'}, # Total Share Prices for All Shares for the United States
    {'series_name': 'Population', 'series_id': 'POP', 'frequency': 'm'}, # US population
    {'series_name': 'US Debt', 'series_id': 'GFDEGDQ188S', 'frequency': 'q'}, # US Debt
    {'series_name': 'US Trade Balance', 'series_id': 'TB3MS', 'frequency': 'm'}, # US Trade Balance
    {'series_name': 'US Bonds Rate 10y', 'series_id': 'DGS10', 'frequency': 'm'}, # US Bonds Rate 10 years
    {'series_name': 'US Bonds Rate 1y', 'series_id': 'DGS1', 'frequency': 'm'}, # US Bonds Rate 1 year
    {'series_name': 'AAA Bond Rate', 'series_id': 'AAA', 'frequency': 'm'}, # AAA Average Corporate Bond Yield
    {'series_name': 'BAA Bond Rate', 'series_id': 'BAA', 'frequency': 'm'}, # BAA Average Corporate Bond Yield
    {'series_name': 'Money Velocity', 'series_id': 'M1V', 'frequency': 'q'}, # Money Velocity (of spending)
    {'series_name': 'GDP per capita', 'series_id': 'A939RX0Q048SBEA', 'frequency': 'q'}, # GDP per capita
    {'series_name': 'Credit Card Transactions', 'series_id': 'CCSA', 'frequency': 'm'}, # Credit Card Transactions
    {'series_name': 'PMI Manufacturing', 'series_id': 'MANEMP', 'frequency': 'm'}, # Manufacturing Employment
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
#                                     MAIN CLASS
# -------------------------------------------------------------------------------------------------
class Fred:
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
        tasks = [fred.get_one_series(**one_series) for one_series in self.series]
        # results is an empty dataframe with the same time range defined in the params
        date_range = pd.date_range(start=self.params['observation_start'], end=self.params['observation_end'], freq='MS')
        df_results = pd.DataFrame(index=date_range)
        for task in asyncio.as_completed(tasks): # keeps the order of the results
            df_result = await task
            # results.append(df_result)
            df_results = pd.concat([df_results, df_result], axis=1)
        # as some recent values may not be yet known, we fill the last missing values with the last known value
        df_results.iloc[-10:] = df_results.iloc[-10:].fillna(method='ffill')
        return df_results

# *************************************************************************************************
#                               STARTS HERE -
# -------------------------------------------------------------------------------------------------

fred = Fred(series, start_date, end_date)
# get all the results in a dataframe
df_results = loop.run_until_complete(fred.get_api_results())
df_results.sort_index(axis=1, inplace=True)
df_results['S&P500-RSI'] = RSI(df_results['S&P500'], timeperiod=14)

print(df_results)
print(f"Total time elapsed: {time.perf_counter() - timer_start:.2f} seconds")




# Compute the autocorrelation matrix of the DataFrame
corr = df_results.corr()

ax = sns.heatmap(corr, 
            xticklabels=corr.columns, 
            yticklabels=corr.columns, 
            cmap='coolwarm', 
            annot=True, 
            fmt='.2f', 
            vmin=-1, 
            vmax=1,
            mask=np.triu(np.ones_like(corr, dtype=np.bool)),
            annot_kws={"size": 6}, # make text smaller
            cbar_kws={'label': 'Correlation'})

# Show the plot
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
ax.set_title("Autocorrelation Heatmap", fontsize=10)
ax.set_xlabel("X-axis", fontsize=8)
ax.set_ylabel("Y-axis", fontsize=8)
plt.show()