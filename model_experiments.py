import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load train dataset
df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')

features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']
X = df_train[features].copy()
y = df_train['overall_yield'].values

# Feature engineering
for df in [X]:
    df['residence_time'] = df['length_m'] / df['flow_rate_L_min']
    df['temp_diff'] = df['jacket_temperature_K'] - df['inlet_temperature_K']
    df['inv_temp_inlet'] = 1.0 / df['inlet_temperature_K']
    df['inv_temp_jacket'] = 1.0 / df['jacket_temperature_K']

all_features = list(X.columns)

# We will evaluate different models using 5-Fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model_name, model_fn):
    rmses = []
    for train_idx, val_idx in cv.split(X):
        X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[all_features])
        X_val_scaled = scaler.transform(X_val[all_features])
        
        # Train and predict
        y_pred = model_fn(X_train_scaled, y_train, X_val_scaled)
        
        # Clip predictions to [0, 100]
        y_pred = np.clip(y_pred, 0.0, 100.0)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)
    print(f"{model_name} CV RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")

# 1. Ridge Regressor
def run_ridge(X_tr, y_tr, X_va):
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    return model.predict(X_va)

evaluate_model("Ridge Regression", run_ridge)

# 2. SVR (Support Vector Regressor)
def run_svr(X_tr, y_tr, X_va):
    model = SVR(C=10.0, epsilon=0.1)
    model.fit(X_tr, y_tr)
    return model.predict(X_va)

evaluate_model("Support Vector Regression (SVR)", run_svr)

# 3. Random Forest Regressor
def run_rf(X_tr, y_tr, X_va):
    model = RandomForestRegressor(random_state=42, max_depth=4, n_estimators=100)
    model.fit(X_tr, y_tr)
    return model.predict(X_va)

evaluate_model("Random Forest Regressor (depth=4)", run_rf)

# 4. Hurdle Model (Classifier + Regressor)
# Stage 1: Classify zero vs non-zero
# Stage 2: Regressor on non-zero rows only
def run_hurdle(X_tr, y_tr, X_va):
    # Classifier
    y_tr_zero = (y_tr == 0.0).astype(int)
    clf = RandomForestClassifier(random_state=42, max_depth=3, n_estimators=50)
    clf.fit(X_tr, y_tr_zero)
    is_zero_preds = clf.predict(X_va) # 1 if predicted zero, 0 if predicted non-zero
    
    # Regressor trained on non-zero only
    non_zero_mask = (y_tr > 0.0)
    X_tr_nonzero = X_tr[non_zero_mask]
    y_tr_nonzero = y_tr[non_zero_mask]
    
    reg = SVR(C=10.0, epsilon=0.1)
    reg.fit(X_tr_nonzero, y_tr_nonzero)
    
    y_pred = reg.predict(X_va)
    y_pred[is_zero_preds == 1] = 0.0
    return y_pred

evaluate_model("Hurdle Model (RF Classifier + SVR Regressor)", run_hurdle)
