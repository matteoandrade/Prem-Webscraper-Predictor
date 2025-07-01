import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
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

print("Accuracy score:", acc, "Prediction score:", prec)