import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
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

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "MLP": MLPRegressor(random_state=42, max_iter=2000)
}

params = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5, 6, None],
        "min_samples_split": [2, 5, 10]
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.8, 1.0]
    },
    "Extra Trees": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5, 6, None],
        "min_samples_split": [2, 5, 10]
    },
    "Hist Gradient Boosting": {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "max_iter": [50, 100, 150]
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "gamma": ['scale', 'auto', 0.01, 0.1, 1.0],
        "epsilon": [0.01, 0.1, 1.0]
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.001, 0.01]
    }
}

for name, model in models.items():
    print(f"\nTuning {name}...")
    
    rmses = []
    best_params_list = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[all_features])
        X_val_scaled = scaler.transform(X_val[all_features])
        
        grid = GridSearchCV(model, params[name], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val_scaled)
        y_pred = np.clip(y_pred, 0.0, 100.0)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)
        best_params_list.append(grid.best_params_)
        
    print(f"{name} CV RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print("Sample Best Parameters:", best_params_list[0])
