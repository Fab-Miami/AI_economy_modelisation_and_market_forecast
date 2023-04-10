# # conda activate py38

# import requests
# import json

# quandl_api_key = "1PeNtzRGtmacF-LgsG6S"

# start_date = '1970-01-01'
# end_date = '1970-12-31'

# url = f"https://www.quandl.com/api/v3/datatables/SHARADAR/SF1?dimension=MRY&ticker=DJIA&date.gte={start_date}&date.lte={end_date}&api_key={quandl_api_key}"

# # url = f"https://data.nasdaq.com/api/v3/datasets/WIKI/AAPL.json?start_date={start_date}&end_date={end_date}&order=asc&column_index=4&collapse=quarterly&transformation=rdiff"
# response = requests.get(url)
# dow_data = json.loads(response.text)

# print(dow_data)


# import quandl
# import pandas as pd

# # Replace 'your_api_key' with your actual API key
# quandl.ApiConfig.api_key = '1PeNtzRGtmacF-LgsG6S'

# # Download the Dow Jones Industrial Average data
# dow_data = quandl.get('BCB/UDJIAD1', start_date='1970-01-01', end_date='2023-12-31')

# # Save the data to a CSV file
# dow_data.to_csv('dow_data.csv')

# # Print the data
# print(dow_data)


import yfinance as yf
import pandas as pd

# Download the Dow Jones Industrial Average data
djia = yf.Ticker('^DJI')

# Define the start and end dates
start_date = '1970-01-01'
end_date = '2023-12-31'

# Get historical data, including adjusted close and volume
dow_data = djia.history(start=start_date, end=end_date)


# Print the data
print(dow_data)
