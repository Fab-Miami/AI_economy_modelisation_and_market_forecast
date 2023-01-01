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



import requests
import matplotlib.pyplot as plt
import numpy as np
from fredapi import Fred


# Set the API endpoint and your API key
api_endpoint = "https://api.stlouisfed.org/fred/series/observations"
api_key = "9e28d63eab23f1dea77320c11110fa4b"

fred = Fred(api_key=api_key)

# Get GDP data from FRED
gdp = fred.get_series('GDPC1')
unemployment_rate = fred.get_series('UNRATE')
inflation_rate = fred.get_series('CPIAUCSL') # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
interest_rate = fred.get_series('DFF')
corporate_profits = fred.get_series('CP')
nasdaq = fred.get_series('NASDAQCOM')
nasdaq100 = fred.get_series('NASDAQ100')
snp500 = fred.get_series('SP500')
dowjones = fred.get_series('DJIA')
cci = fred.get_series('CSCICP03USM665S')


# print(gdp)
# print(unemployment_rate)
# print(corporate_profits)
# print("SNP 500 *********************************************")
# print(snp500)
print("CCI *********************************************")
print(cci)


# # Set the series ID for the 30-Year Fixed Rate Mortgage Average
# series_id = "MORTGAGE30US"

# # Set the parameters for the API request
# params = {
#     "api_key": api_key,
#     "series_id": series_id,
#     "observation_start": "2010-01-01",  # start date for the data
#     "observation_end": "2020-01-01",  # end date for the data
#     "units": "lin",  # scale the data as a linear series
#     "frequency": "m",  # retrieve monthly data
#     "file_type": "json",
#     "sort_order": "asc",  # sort the data in ascending order
# }

# # Send the request to the API endpoint
# response = requests.get(api_endpoint, params=params)

# # Check the status code of the response
# if response.status_code != 200:
#     print("Error: API request failed")
# else:
#     # Print the response data
#     print(response.json())
#     pass

# observations = response.json()["observations"]

# dates = [observation["date"] for observation in observations]
# values = [observation["value"] for observation in observations]
# dates_np = np.array(dates, dtype="datetime64")

# # Plot the values as a line chart
# plt.plot(dates_np, values)
# # plt.xticks(dates_np, dates, rotation=45)

# plt.xticks(dates_np[::10], dates[::10], rotation=45, fontsize=8)
# plt.yticks(values[::10])
# plt.grid(axis="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)



# # Add a title and labels to the axes
# plt.title("30-Year Fixed Rate Mortgage Average")
# plt.xlabel("Date")
# plt.ylabel("Mortgage Average (in %)")
# plt.show()
