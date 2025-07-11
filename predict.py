import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read in the data
matches = pd.read_csv("matches.csv", index_col=0)

# Clean/convert the data
matches["date"] = pd.to_datetime(matches["Date"])
matches.drop("Date", axis=1, inplace=True)
matches["venue_num"] = matches["Venue"].astype("category").cat.codes
matches["opp_num"] = matches["Opponent"].astype("category").cat.codes
matches["hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_num"] = matches["date"].dt.day_of_week

def res_pts(result):
    '''
    Returns 3 if the result is a win, 1 if it is a draw, or 0 if it is a loss
    '''
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0
    
# Convert match results to the number of points won
matches["points"] = matches["Result"].apply(res_pts)

print("types:", matches.dtypes)

# Define columns
cols = ["GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt", "xG", "xGA", "Poss"]
roll_cols = [f"{c.lower()}_roll" for c in cols]
predictors = ["venue_num", "opp_num", "hour", "day_num"]

# Compute rolling averages
def rolling_avg(group, col, new_col, window):
    '''
    Computes group's rolling average of the col columns and stores the values in the new_col columns
    '''
    group = group.sort_values("date")
    rolling = group[col].rolling(window, closed="left").mean()
    group[new_col] = rolling
    return group

def apply_rolling_averages(data, cols, window=3):
    '''
    Applies the rolling_avg function to the data DataFrame's cols columns
    '''
    new_cols = [f"{c.lower()}_roll" for c in cols]
    rolling_data = data.sort_values("date").groupby("Team", group_keys=False).apply(lambda x: rolling_avg(x, cols, new_cols, window)).reset_index(drop=True)
    return rolling_data, new_cols

# Get the rolling averages
matches_roll, roll_cols = apply_rolling_averages(matches, cols)

# Split between training and testing data
train = matches_roll[matches_roll["date"] < "2025-01-01"].copy()
test = matches_roll[matches_roll["date"] >= "2025-01-01"].copy()

# Drop rows without rolling data
train = train.dropna(subset=roll_cols)
test = test.dropna(subset=roll_cols)

# Fitting data
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1, class_weight="balanced")
rf.fit(train[predictors + roll_cols], train["points"])

# Making predictions
test["prediction"] = rf.predict(test[predictors + roll_cols])

# Evaluating prediction accuracy
print("Accuracy:", accuracy_score(test["points"], test["prediction"]))
print("\nConfusion Matrix:\n", confusion_matrix(test["points"], test["prediction"]))
print("\nClassification Report:\n", classification_report(test["points"], test["prediction"]))

# Predict probabilities for each class
probs = rf.predict_proba(test[predictors + roll_cols])

# Classes might not be in order [0, 1, 3] ==> map manually
# Get class order from the trained model
class_order = rf.classes_

# Create a mapping of class index to point value
expected_points = sum(probs[:, i] * class_order[i] for i in range(len(class_order)))
test["expected_points"] = expected_points

# Show Sample Predictions
print("\nSample predictions:")
print(test[["Team", "Opponent", "date", "Result", "points", "prediction", "expected_points"]].head(10))

