import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from tools.tool_fct import plot_dataframes

#########################################################################################################
#                                                                                                       #
#   This script creates a composite volatility index (VIX) based on the S&P 500 volatility and the VIX. #
#   The S&P 500 volatility is scaled by a factor of 1500 and then added to 5.                           #
#   The VIX is used after the S&P 500 volatility is available.                                          #
#   Result is saved as csv file in saved_data_from_generators/                                          #
#                                                                                                       #
#########################################################################################################

# Download S&P 500 daily price
sp500_data = yf.download('^GSPC', start='1970-01-01', end='2023-12-31', interval='1d')

# Download VIX daily price
vix_data = yf.download('^VIX', start='1970-01-01', end='2023-12-31', interval='1d')

# Calculate daily returns for the S&P 500
sp500_data['Returns'] = sp500_data['Adj Close'].pct_change()

# Group the daily returns data by month and calculate the standard deviation for each month
monthly_volatility = sp500_data['Returns'].groupby(pd.Grouper(freq='M')).std()

# Create a DataFrame with the month in one column and volatility in the other
monthly_volatility_df = pd.DataFrame({'Month': monthly_volatility.index, 'SP500_Volatility': monthly_volatility.values})

# Resample the VIX data to get the monthly VIX values
monthly_vix = vix_data['Adj Close'].resample('M').last()

# Create a DataFrame with the month in one column and VIX in the other
monthly_vix_df = pd.DataFrame({'Month': monthly_vix.index, 'VIX': monthly_vix.values})

# Merge the DataFrames with an outer join
merged_df = pd.merge(monthly_volatility_df, monthly_vix_df, on='Month', how='outer')

# Scale the SP500_Volatility values
merged_df['SP500_Volatility'] = (merged_df['SP500_Volatility'] * 1500) + 5

# Find the index where the VIX data starts to be available
vix_start_index = merged_df.loc[~merged_df['VIX'].isna()].index[0]

# Create a new column 'Composite_VIX' with the scaled SP500_Volatility before VIX is available, and VIX after that
merged_df['Composite_VIX'] = merged_df.apply(lambda row: row['SP500_Volatility'] if pd.isna(row['VIX']) or row.name < vix_start_index else row['VIX'], axis=1)

# Set the index and change it to the first day of the month & rename it to 'Date'
merged_df.set_index(pd.to_datetime(merged_df['Month']).dt.to_period('M').dt.to_timestamp(), inplace=True)
merged_df.index.name = 'Date'
merged_df.drop(columns=['SP500_Volatility', 'VIX', 'Month'], inplace=True)

# Save to csv
merged_df.to_csv('saved_data_from_generators/composite_vix.csv')


### Plot the Composite_VIX column ###
# plt.figure(figsize=(10, 6))
# plt.plot(merged_df['Composite_VIX'])
# plt.xlabel('Month')
# plt.ylabel('Composite VIX')
# plt.title('Composite VIX (Scaled SP500 Volatility and VIX)')
# plt.show()