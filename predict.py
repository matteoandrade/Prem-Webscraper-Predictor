import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
# ^^^ imports ^^^

def rolling_avg(group, col, new_col):
    '''
    Computes group's rolling average of the col columns and stores the values in the new_col columns
    '''
    # Sort by date
    group = group.sort_values("date")

    # Get rolling average
    rolling = group[col].rolling(3, closed="left").mean()
    group[new_col] = rolling

    # Drop empty values
    group = group.dropna(subset=new_col)

    return group

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

# Breaking up data
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < "2025-01-01"]
test  = matches[matches["date"] >= "2025-01-01"]
predictors = ["venue_num", "opp_num", "hour", "day_num"]

# Fitting data
rf.fit(train[predictors], train["obj"])
preds = rf.predict(test[predictors])

# Testing accuracy
acc = accuracy_score(test["obj"], preds)
combined = pd.DataFrame(dict(actual=test["obj"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])
prec = precision_score(test["obj"], preds)

# # Breaking up by team
# grouped = matches.groupby("Team")
# group = grouped.get_group("Liverpool")

# Computing rolling averages
cols = ["GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt", "xG", "xGA"]
new_cols = [f"{c.lower()}_roll" for c in cols]
matches_roll = matches.groupby("Team").apply(lambda x: rolling_avg(x, cols, new_cols))
matches_roll = matches_roll.droplevel("Team")
matches_roll.index = range(matches_roll.shape[0])

def predict(data, predictors):
    """
    Makes predictions based off the predictors in the data DataFrame
    """

    # Determining training and testing dates
    train = data[data["date"] <  "2025-01-01"]
    test  = data[data["date"] >= "2025-01-01"]

    # Fitting data
    rf.fit(train[predictors], train["obj"])
    preds = rf.predict(test[predictors])

    # Testing precision of prediction
    combined = pd.DataFrame(dict(actual=test["obj"], prediction=preds), index=test.index)
    pd.crosstab(index=combined["actual"], columns=combined["prediction"])
    prec = precision_score(test["obj"], preds)

    return combined, prec

# Predicting matches based off the desired data
comb, prec = predict(matches_roll, predictors+cols)
comb = comb.merge(matches_roll[["date", "Team", "Opponent", "Result"]], left_index=True, right_index=True)

# Dictionary to map team names
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_val = {
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham"
}

# Cleaning up team names
mapping = MissingDict(**map_val)
comb["new_team"] = comb["Team"].map(mapping)

# Merging the table with itself
merged = comb.merge(combined, left_on=["date", "new_team", ], right_on=["date", "Opponent"])

print("Prediction score:", prec)
print("Combined:\n", comb)