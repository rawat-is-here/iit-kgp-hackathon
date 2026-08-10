import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df_150 = pd.read_csv('train_dataset.csv')
features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']

def simulate_pfr(params, F, C0, T_in, L, T_j, N_steps=30):
    log_alpha, log_A1, E1, log_A2, E2, log_scale, order_a, order_b = params
    alpha = np.exp(log_alpha)
    A1 = np.exp(log_A1)
    A2 = np.exp(log_A2)
    scale = np.exp(log_scale)
    
    num_rows = len(F)
    CA = C0.copy()
    CB = np.zeros(num_rows)
    tau = L / F
    dt = tau / N_steps
    
    for step in range(N_steps):
        t = step * dt
        T = T_j - (T_j - T_in) * np.exp(-alpha * t)
        T = np.clip(T, 100.0, 1000.0)
        
        k1 = A1 * np.exp(-E1 / T)
        k2 = A2 * np.exp(-E2 / T)
        
        r1 = k1 * (CA ** order_a)
        r2 = k2 * (CB ** order_b)
        
        CA = np.clip(CA - r1 * dt, 0.0, None)
        CB = np.clip(CB + (r1 - r2) * dt, 0.0, None)
        
    return np.clip(scale * (CB / C0) * 100.0, 0.0, 100.0)

def fit_pfr_ode(F_tr, C0_tr, T_in_tr, L_tr, T_j_tr, y_tr):
    def objective(params):
        pred = simulate_pfr(params, F_tr, C0_tr, T_in_tr, L_tr, T_j_tr)
        return np.mean((y_tr - pred) ** 2)
    
    initial_guess = [0.0, 15.0, 5000.0, 18.0, 7000.0, 0.0, 1.0, 1.0]
    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[
        (-5.0, 5.0), (5.0, 30.0), (1000.0, 15000.0), (5.0, 35.0), (1000.0, 18000.0), (-1.0, 1.0),
        (0.2, 3.0), (0.2, 3.0)
    ])
    return res.x

cv = KFold(n_splits=5, shuffle=True, random_state=42)
oof_mlp = np.zeros(len(df_150))
oof_hgb = np.zeros(len(df_150))
y = df_150['overall_yield'].values

for train_idx, val_idx in cv.split(df_150):
    train_fold = df_150.iloc[train_idx].copy()
    val_fold = df_150.iloc[val_idx].copy()
    
    for fold in [train_fold, val_fold]:
        fold['residence_time'] = fold['length_m'] / fold['flow_rate_L_min']
        fold['temp_diff'] = fold['jacket_temperature_K'] - fold['inlet_temperature_K']
        fold['inv_temp_inlet'] = 1.0 / fold['inlet_temperature_K']
        fold['inv_temp_jacket'] = 1.0 / fold['jacket_temperature_K']
        
    cols = features + ['residence_time', 'temp_diff', 'inv_temp_inlet', 'inv_temp_jacket']
    
    F_tr, C0_tr, T_in_tr, L_tr, T_j_tr = (
        train_fold['flow_rate_L_min'].values, train_fold['concentration_mol_L'].values,
        train_fold['inlet_temperature_K'].values, train_fold['length_m'].values, train_fold['jacket_temperature_K'].values
    )
    best_params = fit_pfr_ode(F_tr, C0_tr, T_in_tr, L_tr, T_j_tr, train_fold['overall_yield'].values)
    
    train_fold['y_phys'] = simulate_pfr(best_params, F_tr, C0_tr, T_in_tr, L_tr, T_j_tr)
    val_fold['y_phys'] = simulate_pfr(
        best_params, val_fold['flow_rate_L_min'].values, val_fold['concentration_mol_L'].values,
        val_fold['inlet_temperature_K'].values, val_fold['length_m'].values, val_fold['jacket_temperature_K'].values
    )
    
    all_cols = cols + ['y_phys']
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(train_fold[all_cols])
    X_va_scaled = scaler.transform(val_fold[all_cols])
    
    mlp = MLPRegressor(random_state=42, alpha=0.001, hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=3000)
    mlp.fit(X_tr_scaled, train_fold['overall_yield'].values)
    oof_mlp[val_idx] = np.clip(mlp.predict(X_va_scaled), 0.0, 100.0)
    
    hgb = HistGradientBoostingRegressor(random_state=42, max_depth=3, learning_rate=0.05, max_iter=100)
    hgb.fit(X_tr_scaled, train_fold['overall_yield'].values)
    oof_hgb[val_idx] = np.clip(hgb.predict(X_va_scaled), 0.0, 100.0)

for w_mlp in np.linspace(0.0, 1.0, 11):
    w_hgb = 1.0 - w_mlp
    blend_w = w_mlp * oof_mlp + w_hgb * oof_hgb
    rmse_w = np.sqrt(mean_squared_error(y, blend_w))
    print(f"  Weight MLP={w_mlp:.1f}, HGB={w_hgb:.1f} -> RMSE: {rmse_w:.4f}")
