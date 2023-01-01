import asyncio
import aiohttp
import time
import pandas as pd


class Fred:
    def __init__(self, observation_start, observation_end):
        self.api_key = "9e28d63eab23f1dea77320c11110fa4b"
        self.api_endpoint = "https://api.stlouisfed.org/fred/series/observations"
        self.params = {
            "api_key": self.api_key,
            "observation_start": observation_start,  # start date for the data
            "observation_end": observation_end,  # end date for the data
            "units": "lin",  # scale the data as a linear series
            "file_type": "json",
            "sort_order": "asc",  # sort the data in ascending order
        }

    async def get_series(self, serie_name, series_id, frequency='m'):
        async with aiohttp.ClientSession() as session:

            self.params['series_id'] = series_id
            self.params['frequency'] = frequency
            async with session.get(self.api_endpoint, params=self.params) as response:
                # print(f"Serie ID: {series_id}, Status: {response.status}, Content type: {response.content_type}")
                # print(await response.text())
                try:
                    data = await response.json()
                    df = pd.DataFrame.from_dict(data['observations'])
                    df.set_index('date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                    return df
                except Exception as e:
                    print(f"Serie ID: {series_id}, Status: {response.status}, Content type: {response.content_type}")
                    print(await response.text())
                    print(f"Error: {e}")
                    return None

fred = Fred('1947-01-01', '2022-12-31')

start_time = time.perf_counter()

loop = asyncio.get_event_loop()
series = [
    {'serie_name': 'Gross Domestic Product', 'series_id': 'GDPC1', 'frequency': 'q'},
    {'serie_name': 'Unemployment Rate', 'series_id': 'UNRATE', 'frequency': 'm'},
    {'serie_name': 'Consumer Price Index', 'series_id': 'CPIAUCSL', 'frequency': 'm'}, # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
    {'serie_name': 'Interest Rate', 'series_id': 'DFF', 'frequency': 'm'},
    {'serie_name': 'Corporate Profits', 'series_id': 'CP', 'frequency': 'q'},
    {'serie_name': 'NASDAQ', 'series_id': 'NASDAQCOM', 'frequency': 'm'},
    {'serie_name': 'NASDAQ100', 'series_id': 'NASDAQ100', 'frequency': 'm'},
    {'serie_name': 'S&P500', 'series_id': 'SP500', 'frequency': 'm'},
    {'serie_name': 'Dow Jones', 'series_id': 'DJIA', 'frequency': 'm'},
    {'serie_name': 'Consumer Confidence Index', 'series_id': 'CSCICP03USM665S', 'frequency': 'm'}, # Consumer Confidence Index
]



async def get_results():
    tasks = [fred.get_series(**one_serie) for one_serie in series]
    results = []
    for task in asyncio.as_completed(tasks): # keeps the order of the results
        result = await task
        results.append(result)
    return results

results = loop.run_until_complete(get_results())

for one_serie, result in zip(series, results):
    exec(f"{one_serie['series_id']} = result")

# for each series_id, print the first 5 rows of the data
for one_serie in series:
    print("********************")
    print(f"Series Name: {one_serie['series_id']}")
    print("--------------------")
    exec(f"print({one_serie['series_id']}.head())")


# data = pd.concat([gdp, unemployment_rate, inflation_rate, interest_rate, corporate_profits, nasdaq, nasdaq100, snp500, dowjones, cci], axis=1)

end_time = time.perf_counter()

print(f"Total time elapsed: {end_time - start_time:.2f} seconds")

# print(data.tail())
