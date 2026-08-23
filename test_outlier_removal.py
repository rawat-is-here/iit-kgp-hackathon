import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')

features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']
X = df_train[features].copy()
y = df_train['overall_yield'].copy().values

X['residence_time'] = X['length_m'] / X['flow_rate_L_min']
X['temp_diff'] = X['jacket_temperature_K'] - X['inlet_temperature_K']
X['inv_temp_inlet'] = 1.0 / X['inlet_temperature_K']
X['inv_temp_jacket'] = 1.0 / X['jacket_temperature_K']

all_features = list(X.columns)

outlier_indices = [97, 77, 149, 86, 30, 23, 142, 18, 57, 26]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

for n_remove in [0, 1, 2, 3, 5, 8, 10]:
    to_remove = set(outlier_indices[:n_remove])
    
    et_rmses = []
    mlp_rmses = []
    
    for train_idx, val_idx in cv.split(X):
        clean_train_idx = [idx for idx in train_idx if idx not in to_remove]
        
        X_train, y_train = X.iloc[clean_train_idx].copy(), y[clean_train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[all_features])
        X_val_scaled = scaler.transform(X_val[all_features])
        
        et = ExtraTreesRegressor(random_state=42, max_depth=None, min_samples_split=2, n_estimators=200)
        et.fit(X_train_scaled, y_train)
        et_pred = np.clip(et.predict(X_val_scaled), 0.0, 100.0)
        et_rmses.append(np.sqrt(mean_squared_error(y_val, et_pred)))
        
        mlp = MLPRegressor(random_state=42, alpha=0.001, hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=3000)
        mlp.fit(X_train_scaled, y_train)
        mlp_pred = np.clip(mlp.predict(X_val_scaled), 0.0, 100.0)
        mlp_rmses.append(np.sqrt(mean_squared_error(y_val, mlp_pred)))
        
    print(f"Removed {n_remove} outliers from train folds:")
    print(f"  Extra Trees CV RMSE: {np.mean(et_rmses):.4f} +/- {np.std(et_rmses):.4f}")
    print(f"  MLP CV RMSE:         {np.mean(mlp_rmses):.4f} +/- {np.std(mlp_rmses):.4f}")
