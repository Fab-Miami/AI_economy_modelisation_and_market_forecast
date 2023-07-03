import inspect
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, MACD, BBANDS
from colorama import Fore, Back, Style, init
init(autoreset=True)

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

    print(Fore.RED + "----------------------------------------------================================--------------------------------")
    print("n: ", n)
    print(type(dataframes))
    for df in dataframes.items():
        print(df)

    columns = 2
    rows = (n + columns - 1) // columns  # Round up the division

    # Retrieve variable names
    frame = inspect.currentframe().f_back
    var_names = []
    for var_name, var_val in frame.f_locals.items():
        if id(var_val) in [id(df) for df in dataframes]:
            var_names.append(var_name)

    plt.figure(figsize=(10 * columns, 8 * rows))

    # for i, df in enumerate(dataframes):
    for df_name, df in dataframes.items():
        # plt.subplot(rows, columns, i + 1)
        # plt.title(f'{var_names[i]}')

        print(df)

        for col_name in df.columns:
            data = df[col_name]
            # Scale the data between 0 and 1, but only if data is numeric
            if np.issubdtype(df[col_name].dtype, np.number):
                data_scaled = (data - data.min()) / (data.max() - data.min())
                plt.plot(data_scaled, label=col_name)
            else:
                print(f"{Fore.RED}Skipping column {col_name} because it is not numeric")
        plt.xlabel('Index')
        plt.ylabel('Scaled Values')
        plt.legend()

    plt.tight_layout()  # Adjust subplot spacing
    plt.show()



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