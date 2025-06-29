import pandas as pd
# ^^^ imports ^^^

# Read in the data
matches = pd.read_csv("matches.csv", index_col=0)

# Clean the data
matches["date"] = pd.to_datetime(matches["Date"])
matches.drop("Date", axis=1, inplace=True)

print(matches.head())
print(matches.dtypes)