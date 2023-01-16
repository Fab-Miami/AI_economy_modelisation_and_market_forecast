# conda activate py38

# Gross Domestic Product (GDP): This is a measure of the total economic output of a country. An increase in GDP can be a positive sign for the stock market, as it suggests that the economy is growing.

# Unemployment rate: This is the percentage of the labor force that is actively seeking employment but unable to find it. A lower unemployment rate can be a positive sign for the stock market, as it suggests that the economy is strong and people are able to find jobs.

# Inflation rate: This is the rate at which the general level of prices for goods and services is rising, and subsequently, purchasing power is falling. Central banks attempt to limit inflation, and a low inflation rate is generally seen as a positive sign for the stock market.

# Interest rates: This is the rate at which banks lend money to one another. Changes in interest rates can affect the stock market, as they can influence the cost of borrowing money and the return on investment.

# Corporate earnings: This is the profit that a company generates. Strong corporate earnings can be a positive sign for the stock market, as they suggest that companies are performing well and can potentially pay dividends to shareholders.

# Political events: Political events, such as elections or changes in government policies, can also influence the stock market.


import requests
import matplotlib.pyplot as plt
import numpy as np


# Set the API endpoint and your API key
api_endpoint = "https://api.stlouisfed.org/fred/series/observations"
api_key = "9e28d63eab23f1dea77320c11110fa4b"

# Set the series ID for the 30-Year Fixed Rate Mortgage Average
series_id = "MORTGAGE30US"

# Set the parameters for the API request
params = {
    "api_key": api_key,
    "series_id": series_id,
    "observation_start": "2010-01-01",  # start date for the data
    "observation_end": "2020-01-01",  # end date for the data
    "units": "lin",  # scale the data as a linear series
    "frequency": "m",  # retrieve monthly data
    "file_type": "json",
    "sort_order": "asc",  # sort the data in ascending order
}

# Send the request to the API endpoint
response = requests.get(api_endpoint, params=params)

# Check the status code of the response
if response.status_code != 200:
    print("Error: API request failed")
else:
    # Print the response data
    print(response.json())
    pass

observations = response.json()["observations"]

dates = [observation["date"] for observation in observations]
values = [observation["value"] for observation in observations]
dates_np = np.array(dates, dtype="datetime64")

# Plot the values as a line chart
plt.plot(dates_np, values)
# plt.xticks(dates_np, dates, rotation=45)

plt.xticks(dates_np[::10], dates[::10], rotation=45, fontsize=8)
plt.yticks(values[::10])
plt.grid(axis="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.5)



# Add a title and labels to the axes
plt.title("30-Year Fixed Rate Mortgage Average")
plt.xlabel("Date")
plt.ylabel("Mortgage Average (in %)")
plt.show()
