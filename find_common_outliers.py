import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load train dataset
df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')

features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']
X = df_train[features].copy()
y = df_train['overall_yield'].copy().values

# Feature engineering
X['residence_time'] = X['length_m'] / X['flow_rate_L_min']
X['temp_diff'] = X['jacket_temperature_K'] - X['inlet_temperature_K']
X['inv_temp_inlet'] = 1.0 / X['inlet_temperature_K']
X['inv_temp_jacket'] = 1.0 / X['jacket_temperature_K']

all_features = list(X.columns)

# 5-fold CV to get OOF predictions for ET and MLP
cv = KFold(n_splits=5, shuffle=True, random_state=42)
et_preds = np.zeros(len(df_train))
mlp_preds = np.zeros(len(df_train))

# Best params from tuning
# ET: max_depth=None, min_samples_split=2, n_estimators=200
# MLP: alpha=0.001, hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=2000

for train_idx, val_idx in cv.split(X):
    X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[all_features])
    X_val_scaled = scaler.transform(X_val[all_features])
    
    # Extra Trees
    et = ExtraTreesRegressor(random_state=42, max_depth=None, min_samples_split=2, n_estimators=200)
    et.fit(X_train_scaled, y_train)
    et_preds[val_idx] = np.clip(et.predict(X_val_scaled), 0.0, 100.0)
    
    # MLP
    mlp = MLPRegressor(random_state=42, alpha=0.001, hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=3000)
    mlp.fit(X_train_scaled, y_train)
    mlp_preds[val_idx] = np.clip(mlp.predict(X_val_scaled), 0.0, 100.0)

df_train['et_pred'] = et_preds
df_train['mlp_pred'] = mlp_preds
df_train['et_res'] = y - et_preds
df_train['mlp_res'] = y - mlp_preds
df_train['mean_sq_error'] = (df_train['et_res']**2 + df_train['mlp_res']**2) / 2.0

# Print top 10 common outliers
df_common_outliers = df_train.sort_values(by='mean_sq_error', ascending=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Top 10 rows with largest average prediction errors across both models:")
print(df_common_outliers[['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K', 'overall_yield', 'et_pred', 'mlp_pred', 'mean_sq_error']].head(10))

print(f"\nExtra Trees OOF RMSE: {np.sqrt(mean_squared_error(y, et_preds)):.4f}")
print(f"MLP OOF RMSE: {np.sqrt(mean_squared_error(y, mlp_preds)):.4f}")

# Let's test removing these top outliers and recalculating the RMSE on remaining clean data
for n_remove in [1, 2, 3, 5, 8]:
    clean_indices = df_train.sort_values(by='mean_sq_error', ascending=True).index[:-n_remove]
    X_clean = X.iloc[clean_indices].reset_index(drop=True)
    y_clean = y[clean_indices]
    
    clean_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    clean_et_preds = np.zeros(len(clean_indices))
    clean_mlp_preds = np.zeros(len(clean_indices))
    
    for train_idx, val_idx in clean_cv.split(X_clean):
        X_tr, y_tr = X_clean.iloc[train_idx].copy(), y_clean[train_idx]
        X_va, y_va = X_clean.iloc[val_idx].copy(), y_clean[val_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr[all_features])
        X_va_scaled = scaler.transform(X_va[all_features])
        
        et = ExtraTreesRegressor(random_state=42, max_depth=None, min_samples_split=2, n_estimators=200)
        et.fit(X_tr_scaled, y_tr)
        clean_et_preds[val_idx] = np.clip(et.predict(X_va_scaled), 0.0, 100.0)
        
        mlp = MLPRegressor(random_state=42, alpha=0.001, hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=3000)
        mlp.fit(X_tr_scaled, y_tr)
        clean_mlp_preds[val_idx] = np.clip(mlp.predict(X_va_scaled), 0.0, 100.0)
        
    print(f"\nAfter removing {n_remove} outliers:")
    print(f"  Clean Extra Trees RMSE: {np.sqrt(mean_squared_error(y_clean, clean_et_preds)):.4f}")
    print(f"  Clean MLP RMSE: {np.sqrt(mean_squared_error(y_clean, clean_mlp_preds)):.4f}")
