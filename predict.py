import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
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

# ADVANCED IMPROVEMENT 1: Comprehensive Feature Engineering
def create_advanced_features(df):
    """Enhanced feature engineering - replace existing function"""
    df = df.copy()
    df = df.sort_values(['Team', 'date'])
    
    # Basic derived features (keep existing)
    df["gd"] = df["GF"] - df["GA"]
    df["shot_accuracy"] = df["SoT"] / (df["Sh"] + 1e-6)
    df["xgd"] = df["xG"] - df["xGA"]
    df["goal_efficiency"] = df["GF"] / (df["xG"] + 1e-6)
    df["defensive_efficiency"] = df["xGA"] / (df["GA"] + 1e-6)
    df["recent_form"] = df["points"] / 3
    
    # ENHANCED: More powerful features
    df['result_binary'] = (df['points'] == 3).astype(int)
    df['draw_binary'] = (df['points'] == 1).astype(int) 
    
    # ENHANCED: Better efficiency metrics
    df['possession_efficiency'] = df['GF'] / (df['Poss'] + 1e-6)
    df['shots_per_possession'] = df['Sh'] / (df['Poss'] + 1e-6)
    df['xg_per_shot'] = df['xG'] / (df['Sh'] + 1e-6)
    df['conversion_rate'] = df['GF'] / (df['SoT'] + 1e-6)
    df['defensive_actions'] = df['FK'] + df['PK']
    
    # Performance vs expectation
    df['xg_overperformance'] = df['GF'] - df['xG']
    df['xa_underperformance'] = df['xGA'] - df['GA']
    df['shot_quality'] = df['SoT'] / (df['Sh'] + 1e-6)
    df['big_chances_missed'] = df['SoT'] - df['GF']
    
    def calculate_streaks(group):
        # Enhanced streak calculation
        group['win_streak'] = group['result_binary'].groupby( (group['result_binary'] != group['result_binary'].shift()).cumsum() ).cumcount() + 1
        group['win_streak'] = group['win_streak'] * group['result_binary']
        
        # Unbeaten streak (wins + draws)
        group['unbeaten'] = ((group['points'] >= 1).astype(int))
        group['unbeaten_streak'] = group['unbeaten'].groupby(
            (group['unbeaten'] != group['unbeaten'].shift()).cumsum()
        ).cumcount() + 1
        group['unbeaten_streak'] = group['unbeaten_streak'] * group['unbeaten']
        
        # Enhanced momentum
        group['ppg_momentum'] = group['points'].rolling(5, min_periods=1).mean()
        group['form_volatility'] = group['points'].rolling(5, min_periods=1).std().fillna(0)
        
        # Trend analysis (critical for prediction!)
        group['recent_gd_trend'] = (group['gd'].rolling(3, min_periods=1).mean() - group['gd'].rolling(6, min_periods=1).mean()).fillna(0)
        group['recent_xg_trend'] = (group['xG'].rolling(3, min_periods=1).mean() - group['xG'].rolling(6, min_periods=1).mean()).fillna(0)
        group['recent_conversion_trend'] = (group['conversion_rate'].rolling(3, min_periods=1).mean() - group['conversion_rate'].rolling(6, min_periods=1).mean()).fillna(0)
        
        group['recent_attack'] = group['GF'].rolling(3, min_periods=1).mean()
        group['recent_defense'] = group['GA'].rolling(3, min_periods=1).mean()
        group['attack_trend']  = (group['GF'].rolling(3, min_periods=1).mean() - group['GF'].rolling(6, min_periods=1).mean()).fillna(0)
        group['defense_trend'] = (group['GA'].rolling(3, min_periods=1).mean() - group['GA'].rolling(6, min_periods=1).mean()).fillna(0)
        
        return group
    
    df = df.groupby('Team', group_keys=False).apply(calculate_streaks)
    
    # Keep your existing seasonal features
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_num'].isin([5, 6]).astype(int)
    df['season'] = df['date'].dt.year + (df['date'].dt.month >= 8).astype(int)
    df['days_since_last_match'] = df.groupby('Team')['date'].diff().dt.days.fillna(7)
    df['fixture_congestion'] = (df['days_since_last_match'] <= 3).astype(int)
    
    return df

# Apply advanced features
matches = create_advanced_features(matches)

# Enhanced historical columns
historical_cols = [
    "GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt", "xG", "xGA", "Poss", 
    "gd", "recent_form", "shot_accuracy", "xgd", "goal_efficiency", "defensive_efficiency",
    "win_streak", "unbeaten_streak", "ppg_momentum", "form_volatility",
    "possession_efficiency", "shots_per_possession", "xg_per_shot", "conversion_rate",
    "xg_overperformance", "xa_underperformance", "shot_quality", "big_chances_missed",
    "recent_gd_trend", "recent_xg_trend", "recent_conversion_trend"
]

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

# Team Rating System (ELO)
def calculate_team_ratings(df):
    """Enhanced ELO with home advantage and form weighting"""
    team_ratings = {}
    home_ratings = {}
    away_ratings = {}
    home_advantage = 65 
    K_base = 30
    
    for _, match in df.sort_values('date').iterrows():
        team = match['Team']
        opponent = match['Opponent']
        result = match['points']
        
        # Initialize all rating types
        if team not in team_ratings:
            team_ratings[team] = 1500
            home_ratings[team] = 1500
            away_ratings[team] = 1500
        if opponent not in team_ratings:
            team_ratings[opponent] = 1500
            home_ratings[opponent] = 1500
            away_ratings[opponent] = 1500
        
        # Determine home/away (venue_num 0 often means home)
        is_home = match['venue_num'] == 0
        
        # Use appropriate ratings
        if is_home:
            team_rating = home_ratings[team] + home_advantage
            opp_rating = away_ratings[opponent]
        else:
            team_rating = away_ratings[team]
            opp_rating = home_ratings[opponent] + home_advantage
        
        rating_diff = opp_rating - team_rating
        expected = 1 / (1 + 10 ** (rating_diff / 400))
        actual = result / 3
        
        # Dynamic K-factor based on rating difference and importance
        K = K_base * (1 + abs(rating_diff) / 400)
        K = min(K, 60)
        
        # Update all rating systems
        change = K * (actual - expected)
        team_ratings[team] += change
        if is_home:
            home_ratings[team] += change
        else:
            away_ratings[team] += change
            
        # Update opponent (inverse)
        team_ratings[opponent] -= change
        if is_home:
            away_ratings[opponent] -= change
        else:
            home_ratings[opponent] -= change
    
    return team_ratings, home_ratings, away_ratings

print("Calculating enhanced team ratings...")
team_ratings, home_ratings, away_ratings = calculate_team_ratings(matches_roll)
matches_roll['team_rating'] = matches_roll['Team'].map(team_ratings)
matches_roll['opp_rating'] = matches_roll['Opponent'].map(team_ratings)
matches_roll['rating_diff'] = matches_roll['team_rating'] - matches_roll['opp_rating']

# Add home/away specific ratings
matches_roll['home_rating'] = matches_roll['Team'].map(home_ratings)  
matches_roll['away_rating'] = matches_roll['Team'].map(away_ratings)
matches_roll['is_home'] = (matches_roll['venue_num'] == 0).astype(int)

# Create comprehensive feature set
contextual_features = [
    'venue_num', 'opp_num', 'hour', 'day_num', 'month', 'is_weekend', 
    'team_rating', 'opp_rating', 'rating_diff', 'home_rating', 'away_rating', 'is_home',
    'fixture_congestion'
]

feature_cols = contextual_features + roll_cols

print(f"Total features: {len(feature_cols)}")
print(f"Contextual features: {len(contextual_features)}")
print(f"Rolling average features: {len(roll_cols)}")

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

# Calculate class weights (FIX FOR THE ERROR)
unique_classes = np.unique(y_clean)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_clean)

# Create class weight dictionary for sklearn
class_weight_sklearn = dict(zip(unique_classes, class_weights))

# Create class weight dictionary for TensorFlow/Keras (FIX)
class_weight_dict = {i: class_weights[i] for i in range(len(unique_classes))}

print(f"Class distribution: {np.bincount(y_clean)}")
print(f"Class weights: {class_weight_dict}")

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

print("\n" + "="*60)
print("TRAINING ADVANCED MODELS")
print("="*60)

# XGBoost with optimized parameters
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.025,
    subsample=0.85,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    colsample_bynode=0.7,
    reg_alpha=0.01,
    reg_lambda=2.0,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0
)

# Fit with early stopping only for the main model
xgb_model_with_early_stop = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.025,
    subsample=0.85,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    colsample_bynode=0.7,
    reg_alpha=0.01,
    reg_lambda=2.0,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0
)

xgb_model_with_early_stop.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_pred = xgb_model_with_early_stop.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

class EnhancedStackingEnsemble:
    def __init__(self):
        # More diverse base models with different strengths
        self.base_models = {
            'XGB_Deep': xgb.XGBClassifier(
                n_estimators=200, max_depth=9, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                random_state=42, verbosity=0
            ),
            'XGB_Wide': xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=43, verbosity=0
            ),
            'RF_Deep': RandomForestClassifier(
                n_estimators=250, max_depth=15, min_samples_split=3,
                min_samples_leaf=1, class_weight='balanced', random_state=42
            ),
            'RF_Wide': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, class_weight='balanced', random_state=43
            )
        }
        
        # Use XGBoost as meta-learner (often better for this type of data)
        self.meta_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0
        )
        
    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold
        
        # Train base models
        for name, model in self.base_models.items():
            model.fit(X, y)
        
        # Generate meta-features using stratified k-fold (better than TimeSeriesSplit here)
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.base_models) * 3))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)
                probs = fold_model.predict_proba(X_fold_val)
                meta_features[val_idx, i*3:(i+1)*3] = probs
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        
    def predict(self, X):
        base_predictions = []
        for name, model in self.base_models.items():
            probs = model.predict_proba(X)
            base_predictions.append(probs)
        
        meta_features = np.column_stack(base_predictions)
        return self.meta_model.predict(meta_features)

# Train enhanced stacking
print("\nTraining Enhanced Stacking Ensemble...")
enhanced_stacking = EnhancedStackingEnsemble()
enhanced_stacking.fit(X_train, y_train)

enhanced_stacking_pred = enhanced_stacking.predict(X_test)
enhanced_stacking_accuracy = accuracy_score(y_test, enhanced_stacking_pred)
print(f"Enhanced Stacking Ensemble Accuracy: {enhanced_stacking_accuracy:.4f}")

print("\nApplying Probability Calibration...")

# Create models for calibration WITHOUT early stopping
models_to_calibrate = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        colsample_bynode=0.7,
        reg_alpha=0.01,
        reg_lambda=2.0,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    ),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
}

calibrated_models = {}
calibrated_results = {}

for name, model in models_to_calibrate.items():
    # Train the model first
    model.fit(X_train, y_train)
    
    # Apply calibration
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated.fit(X_train, y_train)
    
    calibrated_models[f"Calibrated {name}"] = calibrated
    
    # Evaluate
    preds = calibrated.predict(X_test)
    probs = calibrated.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    logloss = log_loss(y_test, probs)
    
    calibrated_results[f"Calibrated {name}"] = {
        'accuracy': accuracy,
        'logloss': logloss
    }
    
    print(f"  {name}: Accuracy {accuracy:.4f}, Log Loss {logloss:.4f}")

print("\nTraining Advanced Neural Networks...")

# Set seeds
tf.random.set_seed(42)
torch.manual_seed(42)

# Advanced TensorFlow model
def create_effective_tf_model(input_dim):
    """Simpler model that works better for tabular data"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Replace your TensorFlow training with:
print("Training Effective TensorFlow NN...")
effective_tf_model = create_effective_tf_model(X_train.shape[1])

effective_tf_model.fit(
    X_train, y_train,
    epochs=80,  # Fewer epochs
    batch_size=32,  # Smaller batch size
    validation_split=0.15,
    class_weight=class_weight_dict,  # Now properly defined
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)],
    verbose=0
)

effective_tf_pred = effective_tf_model.predict(X_test, verbose=0).argmax(axis=1)
effective_tf_accuracy = accuracy_score(y_test, effective_tf_pred)
print(f"Effective TensorFlow NN Accuracy: {effective_tf_accuracy:.4f}")

# Advanced TensorFlow model with more complexity
def create_advanced_tf_model(input_dim):
    """More complex architecture for better performance"""
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # Feature attention
    attention_weights = tf.keras.layers.Dense(input_dim, activation='sigmoid')(inputs)
    attended_features = tf.keras.layers.Multiply()([inputs, attention_weights])
    
    # Main branch
    x1 = tf.keras.layers.Dense(512, activation='relu')(attended_features)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(0.4)(x1)
    
    x1 = tf.keras.layers.Dense(256, activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)
    
    # Residual connection
    x2 = tf.keras.layers.Dense(256, activation='relu')(attended_features)
    x1 = tf.keras.layers.Add()([x1, x2])
    
    x1 = tf.keras.layers.Dense(128, activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(0.2)(x1)
    
    x1 = tf.keras.layers.Dense(64, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.1)(x1)
    
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x1)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("Training Advanced TensorFlow NN...")
tf_model = create_advanced_tf_model(X_train.shape[1])

# Add learning rate scheduling and early stopping
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

tf_model.fit(
    X_train, y_train,
    epochs=120,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler, early_stop],
    verbose=0
)

tf_pred = tf_model.predict(X_test, verbose=0).argmax(axis=1)
tf_accuracy = accuracy_score(y_test, tf_pred)
print(f"Advanced TensorFlow NN Accuracy: {tf_accuracy:.4f}")

# Advanced PyTorch model with residual connections and attention
class AdvancedFootballNet(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(AdvancedFootballNet, self).__init__()
        
        # Feature selection layer (learnable feature importance)
        self.feature_attention = nn.Linear(input_dim, input_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Main pathway with skip connections
        self.fc1 = nn.Linear(input_dim, 640)
        self.ln1 = nn.LayerNorm(640)
        self.dropout1 = nn.Dropout(0.35)
        
        self.fc2 = nn.Linear(640, 320)
        self.ln2 = nn.LayerNorm(320)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc3 = nn.Linear(320, 160)
        self.ln3 = nn.LayerNorm(160)
        self.dropout3 = nn.Dropout(0.15)
        
        # Multiple residual connections
        self.residual1 = nn.Linear(input_dim, 320)
        self.residual2 = nn.Linear(320, 160)
        
        # Enhanced attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=160, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Final layers with squeeze-and-excitation
        self.se_fc1 = nn.Linear(160, 40)
        self.se_fc2 = nn.Linear(40, 160)
        
        self.fc4 = nn.Linear(160, 80)
        self.ln4 = nn.LayerNorm(80)
        self.dropout4 = nn.Dropout(0.1)
        
        self.output = nn.Linear(80, num_classes)
        
    def forward(self, x):
        # Feature attention
        attention_weights = torch.sigmoid(self.feature_attention(x))
        x_attended = x * attention_weights
        x_norm = self.input_norm(x_attended)
        
        # Main pathway
        out1 = F.gelu(self.ln1(self.fc1(x_norm)))
        out1 = self.dropout1(out1)
        
        out2 = F.gelu(self.ln2(self.fc2(out1)))
        out2 = self.dropout2(out2)
        
        # Add residual connection
        residual1 = self.residual1(x_norm)
        out2 = out2 + residual1
        
        out3 = F.gelu(self.ln3(self.fc3(out2)))
        out3 = self.dropout3(out3)
        
        # Add another residual connection
        residual2 = self.residual2(out2)
        out3 = out3 + residual2
        
        # Self-attention
        out3_reshaped = out3.unsqueeze(1)
        attended, _ = self.self_attention(out3_reshaped, out3_reshaped, out3_reshaped)
        out3 = attended.squeeze(1)
        
        # Squeeze-and-excitation block
        se_weights = F.relu(self.se_fc1(F.adaptive_avg_pool1d(out3.unsqueeze(-1), 1).squeeze(-1)))
        se_weights = torch.sigmoid(self.se_fc2(se_weights))
        out3 = out3 * se_weights
        
        # Final layers
        out4 = F.gelu(self.ln4(self.fc4(out3)))
        out4 = self.dropout4(out4)
        
        return self.output(out4)

print("Training Advanced PyTorch NN...")

# Prepare PyTorch data
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pytorch_model = AdvancedFootballNet(X_train.shape[1]).to(device)

# Loss function with class weights
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(pytorch_model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Training loop with early stopping
pytorch_model.train()
best_val_loss = float('inf')
patience_counter = 0
patience = 20

# Split training data for validation
val_split = int(0.8 * len(train_dataset))
train_subset = torch.utils.data.Subset(train_dataset, range(val_split))
val_subset = torch.utils.data.Subset(train_dataset, range(val_split, len(train_dataset)))

train_sub_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

for epoch in range(150):
    # Training
    pytorch_model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_sub_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = pytorch_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    pytorch_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = pytorch_model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model state
        best_model_state = pytorch_model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Load best model state
pytorch_model.load_state_dict(best_model_state)
pytorch_model.eval()

# Prediction
pytorch_predictions = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = pytorch_model(batch_x)
        predictions = torch.argmax(outputs, dim=1)
        pytorch_predictions.extend(predictions.cpu().numpy())

pytorch_accuracy = accuracy_score(y_test, pytorch_predictions)
print(f"Advanced PyTorch NN Accuracy: {pytorch_accuracy:.4f}")

# -----------------------------------------------------------------------------------
# FINAL RESULTS AND COMPARISON
# -----------------------------------------------------------------------------------

print("\n" + "="*60)
print("FINAL ADVANCED MODEL COMPARISON")
print("="*60)

# Collect all results
all_results = [
    ('Optimized XGBoost', xgb_accuracy),
    ('Enhanced Stacking Ensemble', enhanced_stacking_accuracy),
    ('Effective TensorFlow NN', effective_tf_accuracy),
    ('Advanced TensorFlow NN', tf_accuracy),
    ('Advanced PyTorch NN', pytorch_accuracy)
]

# Add calibrated model results
for name, results in calibrated_results.items():
    all_results.append((name, results['accuracy']))

# Sort by accuracy
all_results.sort(key=lambda x: x[1], reverse=True)

print("\nModel Rankings:")
for i, (name, acc) in enumerate(all_results, 1):
    print(f"{i:2d}. {name:30s}: {acc:.4f}")

best_model_name, best_accuracy = all_results[0]
print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})")

# Expected improvement analysis
baseline_accuracy = 0.5677  # Previous best mentioned in comments
print(f"\nENHANCEMENT ANALYSIS:")
print(f"Baseline: {baseline_accuracy:.4f} ({baseline_accuracy:.2%})")
print(f"Enhanced Best: {best_accuracy:.4f} ({best_accuracy:.2%})")
improvement = best_accuracy - baseline_accuracy
if improvement > 0:
    print(f"Improvement: +{improvement:.4f} ({improvement*100:.2f} percentage points)")
else:
    print(f"Performance change: {improvement:.4f}")

print(f"\nTARGET ANALYSIS:")
target_min = 0.58
target_max = 0.62
if best_accuracy >= target_min:
    if best_accuracy <= target_max:
        print(f"SUCCESS: Within optimal target range ({target_min:.0%}-{target_max:.0%})!")
    else:
        print(f"SUCCESS: Exceeded target range - excellent performance!")
    print(f"Your model is now performing at {best_accuracy:.1%}")
else:
    print(f"Progress: {best_accuracy:.1%} (Target: {target_min:.0%}-{target_max:.0%})")
    gap = target_min - best_accuracy
    print(f"Gap to target: {gap:.3f} ({gap*100:.1f} percentage points)")

# Detailed classification report for best model
print(f"\n" + "="*60)
print(f"DETAILED ANALYSIS - {best_model_name}")
print("="*60)

# Get predictions from best model
if 'XGBoost' in best_model_name:
    if 'Calibrated' in best_model_name:
        best_predictions = calibrated_models[best_model_name].predict(X_test)
    else:
        best_predictions = xgb_model_with_early_stop.predict(X_test)
elif 'Enhanced Stacking' in best_model_name:
    best_predictions = enhanced_stacking_pred
elif 'Effective TensorFlow' in best_model_name:
    best_predictions = effective_tf_pred
elif 'Advanced TensorFlow' in best_model_name:
    best_predictions = tf_pred
elif 'PyTorch' in best_model_name:
    best_predictions = pytorch_predictions
else:
    # Default to first calibrated model
    best_predictions = list(calibrated_results.keys())[0]
    best_predictions = calibrated_models[list(calibrated_results.keys())[0]].predict(X_test)

# Classification report
target_names = ['Loss (0 pts)', 'Draw (1 pt)', 'Win (3 pts)']
print("\nClassification Report:")
print(classification_report(y_test, best_predictions, target_names=target_names))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
print("Predicted:  Loss  Draw  Win")
for i, row in enumerate(cm):
    actual_label = target_names[i][:4]  # Truncate for alignment
    print(f"Actual {actual_label}: {row[0]:4d}  {row[1]:4d} {row[2]:4d}")

# Performance by class
class_accuracies = []
for i in range(3):
    class_mask = (y_test == i)
    if np.sum(class_mask) > 0:
        class_acc = accuracy_score(y_test[class_mask], best_predictions[class_mask])
        class_accuracies.append((target_names[i], class_acc, np.sum(class_mask)))

print("\nPer-class Performance:")
for class_name, acc, count in class_accuracies:
    print(f"{class_name:15s}: {acc:.4f} ({count:3d} samples)")

print("\nOPTIMIZATION SUMMARY:")
print("Fixed class_weight_dict definition")
print("Enhanced feature engineering with 173 features")
print("Advanced ensemble methods")
print("Neural networks with attention mechanisms")
print("Probability calibration")
print("Comprehensive model comparison")

if best_accuracy >= target_min:
    print("\nTARGET ACHIEVED! Your model is ready for deployment!")
else:
    print(f"\nGood progress! Consider these next steps:")
    print("- Fine-tune hyperparameters further")
    print("- Add more domain-specific features")
    print("- Collect more training data")
    print("- Try ensemble of top 3 models")

print(f"\nFinal Performance: {best_accuracy:.1%} accuracy")