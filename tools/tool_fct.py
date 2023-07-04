import math
import inspect
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, MACD, BBANDS
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

def plot_columns_scaled(df, column_list=[]):
    if len(column_list) == 0:
        column_list =[]
        for column in df.columns:
            column_list.append(column)
        col_names = column_list
    else:
        col_names = column_list

    plt.figure(figsize=(10, 8))
    for col_name in df.columns:
        data = df[col_name]
        # Scale the data between 0 and 1
        data_scaled = (data - data.min()) / (data.max() - data.min())
        plt.plot(data_scaled, label=col_name)
    plt.xlabel('Index')
    plt.ylabel('Scaled Values')
    # set the size of the plt
    plt.legend()
    plt.show()


def plot_dataframes(dataframes):
    n = len(dataframes)

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

    plt.tight_layout()  # Ensure subplots do not overlap
    plt.show()


def normalize_dataframe(df):
    for col_name in df.columns:
        # Check if data is numeric
        if np.issubdtype(df[col_name].dtype, np.number):
            max_val = df[col_name].max()
            min_val = df[col_name].min()
            # Check if all values are the same
            if max_val != min_val:
                # Scale the data between 0 and 1
                df[col_name] = (df[col_name] - min_val) / (max_val - min_val)
            else:
                df[col_name] = 0
        else:
            console.print(f"Skipping column {col_name} because it is not numeric", style="bold red")
    return df



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
                mask=np.triu(np.ones_like(corr, dtype=np.bool)),
                annot_kws={"size": 6}, # make text smaller
                cbar_kws={'label': 'Correlation'})

    # Show the plot
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1, top=.9)
    # plt.subplots(figsize=(10,8))

    ax.set_title("Autocorrelation Heatmap", fontsize=10)
    ax.set_xlabel("X-axis", fontsize=8)
    ax.set_ylabel("Y-axis", fontsize=8)
    plt.show()