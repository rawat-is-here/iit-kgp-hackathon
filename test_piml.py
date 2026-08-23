import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('trainer_new.csv')

features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']
X = df[features].copy()
y = df['overall_yield'].copy().values

X['residence_time'] = X['length_m'] / X['flow_rate_L_min']
X['temp_diff'] = X['jacket_temperature_K'] - X['inlet_temperature_K']
X['inv_temp_inlet'] = 1.0 / X['inlet_temperature_K']
X['inv_temp_jacket'] = 1.0 / X['jacket_temperature_K']

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def predict_yield_phys(params, F, T_in, L, T_j):
    log_alpha, log_A1, E1, log_A2, E2, log_scale = params
    alpha = np.exp(log_alpha)
    A1 = np.exp(log_A1)
    A2 = np.exp(log_A2)
    scale = np.exp(log_scale)
    
    tau = L / F
    T_eff = T_j - (T_j - T_in) * np.exp(-alpha * tau)
    T_eff = np.clip(T_eff, 100.0, 1000.0)
    
    k1 = A1 * np.exp(-E1 / T_eff)
    k2 = A2 * np.exp(-E2 / T_eff)
    
    diff = k2 - k1
    diff = np.where(np.abs(diff) < 1e-5, 1e-5 * np.sign(diff), diff)
    
    pred = scale * (k1 / diff) * (np.exp(-k1 * tau) - np.exp(-k2 * tau)) * 100.0
    return np.clip(pred, 0.0, 100.0)

def fit_physical_model(F_tr, T_in_tr, L_tr, T_j_tr, y_tr):
    def objective(params):
        pred = predict_yield_phys(params, F_tr, T_in_tr, L_tr, T_j_tr)
        return np.mean((y_tr - pred) ** 2)
    
    initial_guess = [0.0, 15.0, 5000.0, 18.0, 7000.0, 0.0]
    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[
        (-5.0, 5.0), (5.0, 30.0), (1000.0, 15000.0), (5.0, 35.0), (1000.0, 18000.0), (-1.0, 1.0)
    ])
    return res.x

phys_cv_preds = np.zeros(len(df))
piml_et_preds = np.zeros(len(df))
piml_mlp_preds = np.zeros(len(df))

for train_idx, val_idx in cv.split(X):
    X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
    
    F_tr, T_in_tr, L_tr, T_j_tr = X_train['flow_rate_L_min'].values, X_train['inlet_temperature_K'].values, X_train['length_m'].values, X_train['jacket_temperature_K'].values
    best_params = fit_physical_model(F_tr, T_in_tr, L_tr, T_j_tr, y_train)
    
    y_train_phys = predict_yield_phys(best_params, F_tr, T_in_tr, L_tr, T_j_tr)
    
    F_va, T_in_va, L_va, T_j_va = X_val['flow_rate_L_min'].values, X_val['inlet_temperature_K'].values, X_val['length_m'].values, X_val['jacket_temperature_K'].values
    y_val_phys = predict_yield_phys(best_params, F_va, T_in_va, L_va, T_j_va)
    phys_cv_preds[val_idx] = y_val_phys
    
    X_train_piml = X_train.copy()
    X_val_piml = X_val.copy()
    X_train_piml['y_phys'] = y_train_phys
    X_val_piml['y_phys'] = y_val_phys
    
    all_features = list(X_train_piml.columns)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_piml[all_features])
    X_val_scaled = scaler.transform(X_val_piml[all_features])
    
    et = ExtraTreesRegressor(random_state=42, max_depth=None, min_samples_split=2, n_estimators=200)
    et.fit(X_train_scaled, y_train)
    piml_et_preds[val_idx] = np.clip(et.predict(X_val_scaled), 0.0, 100.0)
    
    mlp = MLPRegressor(random_state=42, alpha=0.001, hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=3000)
    mlp.fit(X_train_scaled, y_train)
    piml_mlp_preds[val_idx] = np.clip(mlp.predict(X_val_scaled), 0.0, 100.0)

print(f"Pure Physical Kinetics CV RMSE:        {np.sqrt(mean_squared_error(y, phys_cv_preds)):.4f}")
print(f"Physics-Informed Extra Trees CV RMSE:  {np.sqrt(mean_squared_error(y, piml_et_preds)):.4f}")
print(f"Physics-Informed MLP CV RMSE:          {np.sqrt(mean_squared_error(y, piml_mlp_preds)):.4f}")
