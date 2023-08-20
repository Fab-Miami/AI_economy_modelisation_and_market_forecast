import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#
from rich import print
from rich.console import Console
console = Console()
#

# define transformations
def calc_pct_change(df):
    # return df
    return df.pct_change().dropna()


def calc_diff(df):
    return df.diff().dropna()


def test_categorization(data_set, macroeconomic_features, technical_features, market_features, political_features):
    all_categorized_features = set(macroeconomic_features + technical_features + market_features + political_features)
    all_data_set_features = set(data_set.columns)
    
    # If any feature from the dataset is missing in the categorized features, print an error.
    if not all_data_set_features.issubset(all_categorized_features):
        missing_features = all_data_set_features.difference(all_categorized_features)
        print(f"[bold red]The following features from the dataset are not categorized:[/bold red] [white]{missing_features}[/white]")
    # If any feature in the categorized features is not in the dataset, print an error.
    elif not all_categorized_features.issubset(all_data_set_features):
        extra_features = all_categorized_features.difference(all_data_set_features)
        print(f"\n[bold red]/!\ The following categorized features are not found in the dataset:[/bold red] [white]{extra_features}[/white]")
        sys.exit(0)
    else:
        print("[bold green]Features in the dataset are correctly categorized.[/bold green]")


def transform_features(data_set, train_size):
    print("[bold yellow]============> APPLYING TRANSFORMATIONS <============[/bold yellow]")
    # define different groups of features
    macroeconomic_features = ['AAA_Bond_Rate', 'BAA_Bond_Rate', 'Consumer_Confidence_Index', 'Consumer_Price_Index', 
                              'Corporate_Profits', 'GDP', 'GDP per capita', 'Money_Velocity', 
                              'PMI_Manufacturing', 'Population', 'US_Debt', 'US_Market_Cap', 
                              'US_Trade_Balance', 'Unemployment_Rate','US_Bonds_Rate_10y', 'US_Bonds_Rate_1y', 'Credit_Card_Transactions', 'Effective_Rate', 'Market_Stress']
    
    technical_features  = ['DJI-BBlower', 'DJI-BBmiddle', 'DJI-BBupper', 'DJI-MACD', 'DJI-MACDhist', 
                          'DJI-MACDsignal', 'DJI-RSI', 'IXIC-BBlower', 'IXIC-BBmiddle', 'IXIC-BBupper', 
                          'IXIC-MACD', 'IXIC-MACDhist', 'IXIC-MACDsignal', 'IXIC-RSI', 'SPX-BBlower', 
                          'SPX-BBmiddle', 'SPX-BBupper', 'SPX-MACD', 'SPX-MACDhist', 'SPX-MACDsignal', 'SPX-RSI']
    
    market_features = ['DJI_close', 'DJI_volume', 'IXIC_close', 'SPX_close', 'SPX_volume', 'IXIC_volume', 'IXIC_volume', 'Composite_VIX']
    
    political_features = ['House_DEM', 'House_REP', 'President_DEM', 'President_REP', 'Senate_DEM', 'Senate_REP']

    # test I'm not forgetting any NEW column I may add in the future
    test_categorization(data_set, macroeconomic_features, technical_features, market_features, political_features)

    # store last values of train set before transformations
    final_train_values = {
        'macroeconomic_features': data_set[macroeconomic_features].iloc[train_size - 1],
        'technical_features': data_set[technical_features].iloc[train_size - 1],
        'market_features': data_set[market_features].iloc[train_size - 1]
    }

    # print("\nType(final_train_values): ", type(final_train_values))
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print("train_size: ", train_size)
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print("data_set[market_features]", data_set[market_features])
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print("\nfinal_train_values: ", final_train_values['market_features'])
    # print("\nfinal_train_values['macroeconomic_features']: ", final_train_values['macroeconomic_features'])
    # print("\nfinal_train_values['macroeconomic_features']['Market_Stress']: ", final_train_values['macroeconomic_features']['Market_Stress'])

    # apply transformations
    data_set_macro = calc_pct_change(data_set[macroeconomic_features])
    # OR
    # data_set_macro = calc_diff(data_set[macroeconomic_features])
    #
    data_set_technical = calc_pct_change(data_set[technical_features])
    # OR
    # data_set_technical = calc_diff(data_set[technical_features])
    #
    data_set_market = calc_pct_change(data_set[market_features])
    # OR
    # data_set_market = calc_diff(data_set[market_features])
    #
    data_set_political = data_set[political_features] # No transformations

    # merge back together
    data_set_transformed = pd.concat([data_set_macro, data_set_technical, data_set_market, data_set_political], axis=1)

    print(f"[bold green]Transformations applied\n\n[/bold green]")
    
    return data_set_transformed.dropna(), final_train_values  # drop rows with NaN values


