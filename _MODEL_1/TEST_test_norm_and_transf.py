from MAIN_building_input_df import get_static_data
from tools.tool_fct import *
from tools.lstm_V1 import *
from tools.lstm_V2 import *
from tools.transformations import *

# #########################################################
#
#   LET'S TEST THE NORMALIZATION AND TRANSFORMATION
#
# #########################################################

df = get_static_data()

# drop some raws
df.drop(columns=['IXIC_volume'], inplace=True)
df.drop(columns=['IXIC_close'], inplace=True)
df.drop(columns=['DJI_volume'], inplace=True)
df.drop(columns=['DJI_close'], inplace=True)


# keep only the 20 first values
df = df.iloc[:20, :]
print("\nlen(df): ", len(df))
print(df.head(100)) # showing there are only 20 rows
# plot
ask_to_plot("Do you want to plot the ORIGINAL data? (yes/no):", {'df_static': df}, normalize=True)

# get the initial values
initial_values = df.iloc[0]
print("\ninitial_values: ", initial_values)                                         

# apply Transformations
df_transformed = calc_pct_change(df)
print("\nlen(df_transformed): ", len(df_transformed))
# plot
ask_to_plot("Do you want to plot all the TRANSFORMED data? (yes/no):", {'df_static Transformed': df_transformed}, normalize=False)

# normalize the dataframe
df_transformed_norm, max_values, min_values = normalize_dataframe(df_transformed)
print("\nmax_values: ", max_values)
print("min_values: ", min_values)
print("len(df_transformed_norm): ", len(df_transformed_norm))
# plot
ask_to_plot("Do you want to plot all the TRANSFORMED and NORMALIZED data? (yes/no):", {'df_static Transformed and Normalized': df_transformed_norm})

# ---------------------------------------------------------------------------------------------------------------------------

# Inverse normalization
df_inv_norm = (df_transformed_norm * (max_values - min_values)) + min_values
print("\nlen(df_inv_norm): ", len(df_inv_norm))
print("df_inv_norm:\n", df_inv_norm)
ask_to_plot("Do you want to plot all the INVERSE NORMALIZED data? (yes/no):", {'df_static Inverse Normalized': df_inv_norm}, normalize=False)

# Inverse transformations (calc_pct_change)
df_inv_transformed = (df_inv_norm + 1).cumprod() * initial_values
print("\nlen(df_inv_transformed): ", len(df_inv_transformed))
print("df_inv_transformed:\n", df_inv_transformed)
ask_to_plot("Do you want to plot all the INVERSE NORMALIZED & INVERSE TRANSFORMED data? (yes/no):", {'df_static inv_transformed': df_inv_transformed})

# plot original values and inv_transformed ones on the same chart
common_columns = df.columns.intersection(df_inv_transformed.columns)
df_combined = pd.concat([df[common_columns], df_inv_transformed[common_columns]], axis=1)
new_columns = [f'Original_{col}' for col in common_columns] + [f'Reconstructed_{col}' for col in common_columns]
df_combined.columns = new_columns
print("\nlen(df_combined): ", len(df_combined))
print("df_combined:\n", df_combined)
ask_to_plot("Do you want to plot the original and inv_transformed data on the same chart? (yes/no):", {'df_combined': df_combined}, normalize=True)