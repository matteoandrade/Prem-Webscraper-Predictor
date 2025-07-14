import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
print("\nRandom Forest Approach:\n")
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

# Mapping points to indices
points_map = {0: 0, 1: 1, 3: 2}
train["target"] = train["points"].map(points_map)
test["target"] = test["points"].map(points_map)

# Normalize the inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(train[predictors + roll_cols])
X_test = scaler.transform(test[predictors + roll_cols])

y_train = train["target"].values
y_test = test["target"].values

# Build and train neural networks
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, class_weight={0: 1.0, 1: 1.0, 2: 1.0})

# Evaluate model accuracy
print("Tensor Flow Neural Network Approach:")
y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test, y_pred, target_names=["Loss", "Draw", "Win"]))

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Wrap in DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define model
class FootballNet(nn.Module):
    def __init__(self, input_size):
        super(FootballNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# Instantiate model
model = FootballNet(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

# Evaluate model accuracy
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"\nPyTorch Neural Network Approach: {correct / total:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["Loss", "Draw", "Win"]))

# Current Best (7/14/25):

#     L   D   W 
# RF .51 .31 .53
# TF .46 .36 .46
# PT .45 .19 .46