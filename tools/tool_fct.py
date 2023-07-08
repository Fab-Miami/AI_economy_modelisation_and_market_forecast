import math
import inspect
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, MACD, BBANDS
from sklearn.metrics import mean_absolute_error, mean_squared_error
#
from rich import print
from rich.console import Console
console = Console()
#

def plot_columns(df, col_names):
    for col_name in col_names:
        data = df[col_name]
        plt.plot(data, label=col_name)
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show(block=False)


def plot_dataframes(dataframes):
    n = len(dataframes)

    # Special case: only one dataframe
    if n == 1:
        df_name, df = list(dataframes.items())[0]
        console.print(f"[bold cyan]\n Plotting: {df_name.upper()} [/bold cyan]")
        plt.figure(figsize=(10, 8))

        for col_name in df.columns:
            data = df[col_name]
            plt.plot(data, label=col_name)

        plt.xlabel('Index')
        plt.ylabel('Scaled Values')
        plt.legend()

        plt.show()
    else:
        # Calculate number of rows and columns for the subplots
        rows = math.ceil(n / 2)  # Change divisor based on how many plots you want per row
        cols = 2

        # Create figure
        fig, axs = plt.subplots(rows, cols, figsize=(7*cols, 5*rows))

        for i, (df_name, df) in enumerate(dataframes.items()):
            ax = axs[i//cols, i%cols]  # Determine the subplot to draw on

            console.print(f"[bold cyan]\n Plotting: {df_name.upper()} [/bold cyan]")

            for col_name in df.columns:
                data = df[col_name]
                ax.plot(data, label=col_name)

            ax.set_xlabel('Index')
            ax.set_ylabel('Scaled Values')
            ax.legend()

        # If there are fewer plots than subplots, remove the empty subplots
        if n < rows * cols:
            for i in range(n, rows * cols):
                fig.delaxes(axs.flatten()[i])

        plt.tight_layout()  # Ensure subplots do not overlap
        plt.show()



# def normalize_dataframe(df):
#     for col_name in df.columns:
#         # Check if data is numeric
#         if np.issubdtype(df[col_name].dtype, np.number):
#             max_val = df[col_name].max()
#             min_val = df[col_name].min()
#             # Check if all values are the same
#             if max_val != min_val:
#                 # Scale the data between 0 and 1
#                 df[col_name] = (df[col_name] - min_val) / (max_val - min_val)
#             else:
#                 df[col_name] = 0
#         else:
#             console.print(f"Skipping column {col_name} because it is not numeric", style="bold red")
#     return df


def autocorrelation(df_results):
    """ Compute the autocorrelation matrix of the DataFrame """
    corr = df_results.corr()

    plt.figure(figsize=(14, 11))

    ax = sns.heatmap(corr, 
                xticklabels=corr.columns, 
                yticklabels=corr.columns, 
                cmap='coolwarm', 
                annot=True, 
                fmt='.2f', 
                vmin=-1, 
                vmax=1,
                mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot_kws={"size": 6}, # make text smaller
                cbar_kws={'label': 'Correlation'})

    # Show the plot
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1, top=.9)
    # plt.subplots(figsize=(10,8))

    ax.set_title("Autocorrelation Heatmap", fontsize=10)
    ax.set_xlabel("X-axis", fontsize=8)
    ax.set_ylabel("Y-axis", fontsize=8)
    plt.show()


def add_indicators(data_set):
    print("\n\n[bold]====> ADDING INDICATORS TO:[/bold]")
    print(data_set.columns)
    print("[bold]--------------------------------------------[/bold]")

    # add RSI
    data_set['SPX-RSI']  = RSI(data_set['SPX_close'], timeperiod=14)
    data_set['DJI-RSI']  = RSI(data_set['DJI_close'], timeperiod=14)
    data_set['IXIC-RSI'] = RSI(data_set['IXIC_close'], timeperiod=14)

    # add MACD
    data_set['SPX-MACD'], data_set['SPX-MACDsignal'], data_set['SPX-MACDhist'] = MACD(data_set['SPX_close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data_set['DJI-MACD'], data_set['DJI-MACDsignal'], data_set['DJI-MACDhist'] = MACD(data_set['DJI_close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data_set['IXIC-MACD'], data_set['IXIC-MACDsignal'], data_set['IXIC-MACDhist'] = MACD(data_set['IXIC_close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # add Bollinger Bands
    data_set['SPX-BBupper'], data_set['SPX-BBlower'], data_set['SPX-BBmiddle'] = BBANDS(data_set['SPX_close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data_set['DJI-BBupper'], data_set['DJI-BBlower'], data_set['DJI-BBmiddle'] = BBANDS(data_set['DJI_close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data_set['IXIC-BBupper'], data_set['IXIC-BBlower'], data_set['IXIC-BBmiddle'] = BBANDS(data_set['IXIC_close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    return data_set


def normalize_dataframe(df):
    df = df.loc[:, df.nunique() != 1]
    max_values = df.max()
    min_values = df.min()
    normalized_df = (df - min_values) / (max_values - min_values)
    return normalized_df, max_values, min_values


def find_missing_dates(df):
    print("[bold]\nMissing dates:[/bold]")

    # Ensure the dataframe is sorted by date
    df = df.sort_index()

    # Get start and end dates, convert Period to Timestamp if necessary
    start_date = df.index[0].to_timestamp() if isinstance(df.index[0], pd.Period) else df.index[0]
    end_date = df.index[-1].to_timestamp() if isinstance(df.index[-1], pd.Period) else df.index[-1]

    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    # Generate all possible year/month combinations between start and end date
    all_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Extract year and month only
    all_dates = all_dates.to_period('M')

    # Extract year and month from dataframe's index and convert to Period if necessary
    df_dates = df.index.to_period('M') if isinstance(df.index, pd.DatetimeIndex) else df.index

    # Find the difference between the sets of dates to find the missing ones
    missing_dates = np.setdiff1d(all_dates, df_dates)
    list_missing_dates = missing_dates.astype(str).tolist()

    
    if len(list_missing_dates) > 0:
        print(f"[red]{list_missing_dates}[/red]\n")
    else:
        print("[bold green]No missing dates found[/bold green]\n")
    
    return


def normalized_for_plot(df):
    print(f"[bold magenta]=> NORMALIZING JUST FOR PLOT[bold magenta]\n")
    original_max_values = {}
    original_min_values = {}
    dfs_plot = {}
    for name, df in df.items():
        dfs_plot[name], original_max_values[name], original_min_values[name] = normalize_dataframe(df)
    return dfs_plot


def ask_to_plot(message, stuff):
    console.print(message, style="bold yellow")
    plot_choice = input().lower()
    if plot_choice == 'y' or plot_choice == 'yes':
        plot_dataframes(normalized_for_plot(stuff))