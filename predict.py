import pandas as pd
# ^^^ imports ^^^

# Read in the data
matches = pd.read_csv("matches.csv", index_col=0)

# Clean the data
matches["date"] = pd.to_datetime(matches["Date"])
matches.drop("Date", axis=1, inplace=True)

# Make numerical values out of strings
matches["venue_num"] = matches["Venue"].astype("category").cat.codes
matches["opp_num"] = matches["Opponent"].astype("category").cat.codes
matches["hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_num"] = matches["date"].dt.day_of_week
matches["obj"] = (matches["Result"] == 'W').astype("int")

print(matches.head())
print(matches.dtypes)