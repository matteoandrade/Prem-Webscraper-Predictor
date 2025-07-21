import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, cross_val_score
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Read in the data
matches = pd.read_csv("matches_5.csv", index_col=0)
cons_win = [1, 3, 5, 7, 10]

# Clean/convert the data
matches["date"] = pd.to_datetime(matches["Date"])
matches.drop("Date", axis=1, inplace=True)
matches["venue_num"] = matches["Venue"].astype("category").cat.codes
matches["opp_num"] = matches["Opponent"].astype("category").cat.codes
matches["hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_num"] = matches["date"].dt.day_of_week

def res_pts(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0
    
matches["points"] = matches["Result"].apply(res_pts)

# IMPROVEMENT 1: Add more sophisticated features
def create_advanced_features(df):
    """Create advanced features for better prediction"""
    df = df.copy()
    
    # Goal difference rolling averages
    df["gd"] = df["GF"] - df["GA"]
    
    # Form indicators (recent performance)
    df["recent_form"] = df["points"] / 3  # 0, 0.33, 1 for L, D, W
    
    # Shot accuracy
    df["shot_accuracy"] = df["SoT"] / (df["Sh"] + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Expected goal difference
    df["xgd"] = df["xG"] - df["xGA"]
    
    # Efficiency metrics
    df["goal_efficiency"] = df["GF"] / (df["xG"] + 1e-6)
    df["defensive_efficiency"] = df["xGA"] / (df["GA"] + 1e-6)
    
    return df

# Create advanced features
matches = create_advanced_features(matches)

# Enhanced historical columns
historical_cols = ["GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt", "xG", "xGA", "Poss", 
                  "gd", "recent_form", "shot_accuracy", "xgd", "goal_efficiency", "defensive_efficiency"]

def rolling_avg(group, col, new_col, window):
    group = group.sort_values("date")
    rolling = group[col].rolling(window, closed="left", min_periods=1).mean()
    group[new_col] = rolling
    return group

def apply_rolling_averages(data, cols, windows=[3, 5, 10]):
    all_col = []
    res = data.copy()
    
    for w in windows:
        new_cols = [f"{c.lower()}_roll_{w}" for c in cols]
        temp = res.sort_values("date").groupby("Team", group_keys=False).apply(
            lambda x: rolling_avg(x, cols, new_cols, w)
        ).reset_index(drop=True)
        res = temp
        all_col.extend(new_cols)
    return res, all_col

# Get rolling averages
matches_roll, roll_cols = apply_rolling_averages(matches, historical_cols, windows=cons_win)

# IMPROVEMENT 2: Add team strength ratings
def calculate_team_ratings(df):
    """Calculate ELO-like team ratings"""
    team_ratings = {}
    K = 30  # ELO K-factor
    
    for _, match in df.sort_values('date').iterrows():
        team = match['Team']
        opponent = match['Opponent']
        result = match['points']
        
        # Initialize ratings
        if team not in team_ratings:
            team_ratings[team] = 1500
        if opponent not in team_ratings:
            team_ratings[opponent] = 1500
            
        # Expected score based on rating difference
        rating_diff = team_ratings[opponent] - team_ratings[team]
        expected = 1 / (1 + 10 ** (rating_diff / 400))
        
        # Actual score (0, 0.5, 1 for L, D, W)
        actual = result / 3
        
        # Update ratings
        team_ratings[team] += K * (actual - expected)
        team_ratings[opponent] += K * (expected - actual)
    
    return team_ratings

# Calculate team ratings and add as features
team_ratings = calculate_team_ratings(matches_roll)
matches_roll['team_rating'] = matches_roll['Team'].map(team_ratings)
matches_roll['opp_rating'] = matches_roll['Opponent'].map(team_ratings)
matches_roll['rating_diff'] = matches_roll['team_rating'] - matches_roll['opp_rating']

# IMPROVEMENT 3: Add seasonal and contextual features
matches_roll['month'] = matches_roll['date'].dt.month
matches_roll['is_weekend'] = matches_roll['day_num'].isin([5, 6]).astype(int)
matches_roll['season'] = matches_roll['date'].dt.year + (matches_roll['date'].dt.month >= 8).astype(int)

# Create feature set
contextual_features = ['venue_num', 'opp_num', 'hour', 'day_num', 'month', 'is_weekend', 
                      'team_rating', 'opp_rating', 'rating_diff']
feature_cols = contextual_features + roll_cols

print(f"Total features: {len(feature_cols)}")
print("Feature categories:")
print(f"- Contextual features: {len(contextual_features)}")
print(f"- Rolling average features: {len(roll_cols)}")

# Prepare data
X_data = matches_roll[feature_cols].copy()
y_data = matches_roll["points"].map({0: 0, 1: 1, 3: 2})

# Handle missing values
X_data = X_data.fillna(X_data.median())

# Remove any remaining NaNs
valid_idx = ~(X_data.isnull().any(axis=1) | y_data.isnull())
X_clean = X_data[valid_idx]
y_clean = y_data[valid_idx]
dates_clean = matches_roll.loc[valid_idx, "date"]

print(f"Dataset shape after cleaning: {X_clean.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Train/test split
train_idx = dates_clean < "2025-01-01"
test_idx = dates_clean >= "2025-01-01"

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]
y_train = y_clean[train_idx].values
y_test = y_clean[test_idx].values

if X_test.shape[0] == 0:
    split_idx = int(0.8 * len(X_scaled))
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y_clean.iloc[:split_idx].values
    y_test = y_clean.iloc[split_idx:].values

print(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

# -----------------------------------------------------------------------------------
# IMPROVEMENT 4: Hyperparameter tuning for Random Forest
# -----------------------------------------------------------------------------------

print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

# Quick grid search for Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    rf_params,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print(f"Best RF parameters: {rf_grid.best_params_}")

# -----------------------------------------------------------------------------------
# IMPROVEMENT 5: PyTorch Neural Network
# -----------------------------------------------------------------------------------

print("\n" + "="*60)
print("TRAINING PYTORCH MODEL")
print("="*60)

# Set seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Enhanced PyTorch model
class AdvancedFootballNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rates=[0.4, 0.3, 0.2], num_classes=3):
        super(AdvancedFootballNet, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# Prepare PyTorch data
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
pytorch_model = AdvancedFootballNet(X_train.shape[1])

# Loss function with class weights
class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train), 
                           dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer with learning rate scheduling
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Training function
def train_pytorch_model(model, train_loader, criterion, optimizer, scheduler, epochs=100):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model

# Train PyTorch model
print("Training PyTorch Neural Network...")
pytorch_model = train_pytorch_model(pytorch_model, train_loader, criterion, optimizer, scheduler)

# Evaluate PyTorch model
def evaluate_pytorch_model(model, test_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.numpy())
            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

pytorch_pred, pytorch_probs, _ = evaluate_pytorch_model(pytorch_model, test_loader)
pytorch_accuracy = accuracy_score(y_test, pytorch_pred)
print(f"PyTorch Neural Network Accuracy: {pytorch_accuracy:.4f}")

# -----------------------------------------------------------------------------------
# IMPROVEMENT 6: Ensemble methods (including PyTorch)
# -----------------------------------------------------------------------------------

print("\n" + "="*60)
print("TRAINING SKLEARN MODELS")
print("="*60)

# Multiple models for ensemble
ml_models = {
    'Tuned Random Forest': best_rf,
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
}

model_predictions = {}
model_probabilities = {}

for name, model in ml_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    
    model_predictions[name] = pred
    model_probabilities[name] = prob
    
    accuracy = accuracy_score(y_test, pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Add PyTorch to the predictions and probabilities
model_predictions['PyTorch Neural Network'] = pytorch_pred
model_probabilities['PyTorch Neural Network'] = pytorch_probs

# IMPROVEMENT 7: Ensemble voting (including PyTorch)
print(f"\nCreating ensemble (including PyTorch)...")
ensemble_probs = np.mean(list(model_probabilities.values()), axis=0)
ensemble_pred = np.argmax(ensemble_probs, axis=1)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

# -----------------------------------------------------------------------------------
# IMPROVEMENT 8: Enhanced TensorFlow Neural Network
# -----------------------------------------------------------------------------------

print("\n" + "="*60)
print("TRAINING TENSORFLOW MODEL")
print("="*60)

tf.random.set_seed(42)
np.random.seed(42)

# More sophisticated neural network
def create_advanced_nn(input_dim, num_classes=3):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train advanced neural network
print("Training TensorFlow Neural Network...")
advanced_nn = create_advanced_nn(X_train.shape[1])

# Class weights
class_w = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_w[i] for i in range(len(class_w))}

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, verbose=0)

# Train with callbacks
history = advanced_nn.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

nn_pred = advanced_nn.predict(X_test, verbose=0).argmax(axis=1)
nn_probs = advanced_nn.predict(X_test, verbose=0)
nn_accuracy = accuracy_score(y_test, nn_pred)

print(f"TensorFlow Neural Network Accuracy: {nn_accuracy:.4f}")

# Add TensorFlow to predictions
model_predictions['TensorFlow Neural Network'] = nn_pred
model_probabilities['TensorFlow Neural Network'] = nn_probs

# Update ensemble with all models (including both neural networks)
print(f"\nUpdating ensemble with all models...")
ensemble_probs = np.mean(list(model_probabilities.values()), axis=0)
ensemble_pred = np.argmax(ensemble_probs, axis=1)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print(f"Final Ensemble Accuracy (5 models): {ensemble_accuracy:.4f}")

# -----------------------------------------------------------------------------------
# Final Results Comparison
# -----------------------------------------------------------------------------------

print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

all_results = []
for name, pred in model_predictions.items():
    acc = accuracy_score(y_test, pred)
    all_results.append((name, acc))

all_results.append(('Ensemble', ensemble_accuracy))
all_results.append(('TensorFlow Neural Network', nn_accuracy))

# Sort by accuracy
all_results.sort(key=lambda x: x[1], reverse=True)

print("\nModel Rankings:")
for i, (name, acc) in enumerate(all_results, 1):
    print(f"{i}. {name}: {acc:.4f}")

# Best model analysis
best_model_name, best_accuracy = all_results[0]
print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")

# Detailed analysis of best performing model
if best_model_name == 'Ensemble':
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=["Loss", "Draw", "Win"]))
elif best_model_name in ml_models:
    print(f"\n{best_model_name} Classification Report:")
    print(classification_report(y_test, model_predictions[best_model_name], target_names=["Loss", "Draw", "Win"]))

# IMPROVEMENT 8: Feature importance analysis
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance from best tree-based model
if 'Random Forest' in best_model_name or 'Gradient' in best_model_name:
    if 'Random Forest' in best_model_name:
        importance_model = best_rf
    else:
        importance_model = ml_models['Gradient Boosting']
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Feature category analysis
    contextual_importance = feature_importance[
        feature_importance['feature'].isin(contextual_features)
    ]['importance'].sum()
    
    rolling_importance = feature_importance[
        ~feature_importance['feature'].isin(contextual_features)
    ]['importance'].sum()
    
    print(f"\nFeature Category Importance:")
    print(f"Contextual features: {contextual_importance:.3f}")
    print(f"Rolling averages: {rolling_importance:.3f}")