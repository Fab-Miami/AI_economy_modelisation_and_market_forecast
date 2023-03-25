# conda activate py38

import requests
import json

quandl_api_key = "1PeNtzRGtmacF-LgsG6S"

start_date = '1970-01-01'
end_date = '1970-12-31'

url = f"https://www.quandl.com/api/v3/datatables/SHARADAR/SF1?dimension=MRY&ticker=DJIA&date.gte={start_date}&date.lte={end_date}&api_key={quandl_api_key}"

url = f"https://data.nasdaq.com/api/v3/datasets/WIKI/AAPL.json?start_date={start_date}&end_date={end_date}&order=asc&column_index=4&collapse=quarterly&transformation=rdiff"
response = requests.get(url)
dow_data = json.loads(response.text)

print(dow_data)