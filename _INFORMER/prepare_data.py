import sys
import os
import aiohttp
PATH = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import text2art
import time
import asyncio
from datetime import datetime
import pandas as pd # V2.0.3
from tools.tool_fct import *
from tools.transformations import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from art import *
from rich import print
from rich.console import Console
from time import sleep
console = Console()


def create_dataset(QUESTIONS=False, TEST_MONTHS=0):
    timer_start = time.perf_counter()

    df_list = ["fred", "elections", "generator", "static"] # "yahoo" is not used anymore as I'm getting SPX from TadingView as a Static download
    # df_list = ["static"]
    dfs = {}

    for name in df_list:
        func_name    = f"get_{name}_data"
        df_name      = f"df_{name}"
        # Create a dictionary with the name of the dataframe as key and the dataframe as value
        print(f"[bold blue]===> Includes: {df_name}[/bold blue]")
        dfs[df_name] = getattr(sys.modules[__name__], func_name)() # Call the function with the name from df_list ( eg: get_fred_data() )
        set_dates_to_first_of_the_month(dfs[df_name])

    sleep(2)

    # print the dataframes
    for df_name, df in dfs.items():
        print(f"[bold green]\n{text2art(df_name.upper())}[/bold green]")
        print(df)
        find_missing_dates(df) # Checks that every month is present

    # plot
    if QUESTIONS:
        ask_to_plot("Do you want to plot all the data? (yes/no):", dfs)

    # printing missing values
    for name, df in dfs.items():
            print(f"\n[bold red]Missing values in {name}:[/bold red]")
            print(df.isna().sum())

    # merge the dataframes
    print(f"\n[bold yellow]============> MERGING DATAFRAMES <============[bold yellow]")
    dataset = list(dfs.values())[0] # Start with the first dataframe
    dataset.index = pd.to_datetime(dataset.index) # Convert the index to datetime
    dataset.index = dataset.index.to_period('M')  # Convert the index to year-month
    for df in list(dfs.values())[1:]:  # Merge all other dataframes
        df.index = df.index.to_period('M')  # Convert the index to year-month
        dataset = dataset.merge(df, left_index=True, right_index=True, how='inner') # inner join to keep only the common dates
    # convert index back to datetime format with first day of the month
    dataset.index = dataset.index.to_timestamp()

    # Fill the holes: Backfill all columns except "Market_Stress"
    dataset = dataset.apply(lambda col: col.bfill() if col.name != "Market_Stress" else col.fillna(0))

    # save the merged dataframe as a csv file for inspection
    dataset.to_csv("dataset_merged.csv", index=True)
    print("DataSet has been saved for inspection as 'dataset_merged.csv'")

    # Checks that every month is present in the merged dataframe
    find_missing_dates(dataset)

    if QUESTIONS:
        ask_to_plot("Do you want to plot of the MERGED dataframe? (yes/no):", {'dataset': dataset})

    dataset = add_indicators(dataset)

    if QUESTIONS:
        ask_to_plot("Do you want to plot of the MERGED dataframe WITH INDICATORS? (yes/no):", {'dataset': dataset})


    # --- At this point the df is ready to be  normalized ---
    # NOTE: with INFORMER, no need to do a percentage change transformation (yeah!)

    print(f"[bold blue]\n============> NORMALIZING DATAFRAMES <============[bold blue]")

    # TODO: Normalization to try: Min-Max Scaling

    # Normalization using Z-Score
    scaler = StandardScaler()
    dataset_normalized = scaler.fit_transform(dataset)
    dataset_normalized = pd.DataFrame(dataset_normalized, index=dataset.index, columns=dataset.columns)

    dataset_normalized.to_csv("dataset_training.csv", index=True)

    print("Normalized dataSet has been saved as 'dataset_training.csv'\n")

    print("Dataset CHECKS:")
    print("Mean should be close to 0 for all columns:\n", dataset_normalized.mean())
    print("Standard deviation hould be close to 1 for all columns:\n", dataset_normalized.std())

    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.joblib')
    print("[bold blue]\nScaler has been saved as 'scaler.joblib' for future use.[bold blue]")


    # plot
    if QUESTIONS:
        ask_to_plot("\nDo you want to plot of the MERGED and NORMALIZED dataframe? (yes/no):", {'dataset': dataset_normalized}, normalize=False)
    

    # print time elapsed
    print(f"[blue]Total time to prepare data: {time.perf_counter() - timer_start:.2f} seconds\n\n[/blue]")

    return dataset_normalized


# *************************************************************************************************
#                                       ELECTION DATA
# -------------------------------------------------------------------------------------------------
def get_elections_data():
    # US elections results (DEM = 1 ; REP = 2)
    df_elections = pd.read_csv(f'{PATH}/../saved_data_elections/USelections.csv')
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
#                                           FRED
# -------------------------------------------------------------------------------------------------
def get_fred_data():
    file_path = f'{PATH}/../saved_data_from_api/fred_results.csv'
    if os.path.isfile(file_path):
        df_fred = pd.read_csv(file_path, index_col=0)
        df_fred.index = pd.to_datetime(df_fred.index)
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
#                                       GENERATOR DATA
# -------------------------------------------------------------------------------------------------

def get_generator_data():
    file_path = f'{PATH}/../saved_data_from_generators/'
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
    file_path = f'{PATH}/../saved_data_from_static/'
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

def set_dates_to_first_of_the_month(df):
    df.index = df.index.map(lambda x: x.replace(day=1))
    return df