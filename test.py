import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6],
                   'B': [10, 20, 30, 40, 50, 60],
                   'C': [100, 200, 300, 400, 500, 600]})

# Add some null values
df.iloc[2:4, 1] = None
df.iloc[4:6, 2] = None

# Fill null values with the last known value, for the last 10 rows
df.iloc[-10:].fillna(method='ffill', inplace=True)

print(df)