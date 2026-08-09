import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
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

# 5-fold CV to get Out-Of-Fold (OOF) predictions
cv = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df_train))

for train_idx, val_idx in cv.split(X):
    X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[all_features])
    X_val_scaled = scaler.transform(X_val[all_features])
    
    model = RandomForestRegressor(random_state=42, max_depth=5, n_estimators=100)
    model.fit(X_train_scaled, y_train)
    
    oof_preds[val_idx] = model.predict(X_val_scaled)

# Calculate residuals
df_train['oof_pred'] = oof_preds
df_train['residual'] = y - oof_preds
df_train['sq_error'] = df_train['residual'] ** 2

df_outliers = df_train.sort_values(by='sq_error', ascending=False)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Top 15 rows with largest Out-Of-Fold prediction errors:")
print(df_outliers.head(15))
