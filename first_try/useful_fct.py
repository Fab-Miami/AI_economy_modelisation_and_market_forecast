
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from talib import RSI, MACD, BBANDS

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