# conda activate py38

import requests
import matplotlib.pyplot as plt
import numpy as np


# Set the API endpoint and your API key
api_endpoint = "https://api.stlouisfed.org/fred/series/observations"
api_key = "9e28d63eab23f1dea77320c11110fa4b"

# Set the series ID for the 30-Year Fixed Rate Mortgage Average
series_id = "GDP"  # GDPC1

# Set the parameters for the API request
params = {
    "api_key": api_key,
    "series_id": series_id,
    "observation_start": "1950-01-01",  # start date for the data
    "observation_end": "2022-01-01",  # end date for the data
    # "units": "chg",  # set the data to be quarterly percent change from the previous quarter
    "frequency": "q",  # retrieve quarterly data
    "file_type": "json",
    "sort_order": "asc",  # sort the data in ascending order
}

# Send the request to the API endpoint
response = requests.get(api_endpoint, params=params)

# Check the status code of the response
if response.status_code != 200:
    print("\n\n##################### Error: API request failed #####################")
    print(response.json()['error_message'], "\n\n")
else:
    # Print the response data
    print(response.json())
    pass

observations = response.json()["observations"]
for observation in observations:
    print(observation["date"], observation["value"])

dates = [observation["date"] for observation in observations]
values = [observation["value"] for observation in observations]

# dates, values = zip(*[(observation["date"], observation["value"]) for observation in observations])  # same but in one line

dates_np = np.array(dates, dtype="datetime64")

# raw value
values_np = np.array(values, dtype="float64")
#plt.plot(dates_np, values_np, label="(GDP)")

# LOG transformation to stabilize variance
values_np_log = np.log(values_np)
# values_np_log = np.log10(values_np) # log base 10
plt.plot(dates_np, values_np_log, label="Log(GDP)")

# DIFFERENCING to remove trends
values_np_log_diff = np.diff(values_np_log)
plt.plot(dates_np[:-1], values_np_log_diff, label="Differenced Log(GDP)")

# Set plot title and axis labels
plt.title("Real Gross Domestic Product (GDPC1)")
plt.xlabel("Date")
plt.ylabel("Quarterly Percent Change from Previous Quarter")

# Add labels and legend
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()

# Show the plot
plt.show()