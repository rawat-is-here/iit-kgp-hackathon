import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df_150 = pd.read_csv('train_dataset.csv')
features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']

outliers_idx = [97, 77, 149, 86, 30, 23, 142, 18, 57, 26]

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

def run_evaluation(df_train_full, remove_outliers=True, seed=42, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_mlp = np.zeros(len(df_train_full))
    oof_hgb = np.zeros(len(df_train_full))
    
    y = df_train_full['overall_yield'].values
    
    for train_idx, val_idx in cv.split(df_train_full):
        if remove_outliers:
            train_idx_clean = [idx for idx in train_idx if idx not in outliers_idx]
        else:
            train_idx_clean = train_idx
            
        train_fold = df_train_full.iloc[train_idx_clean].copy()
        val_fold = df_train_full.iloc[val_idx].copy()
        
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
        
        mlp = MLPRegressor(random_state=seed, alpha=0.001, hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=3000)
        mlp.fit(X_tr_scaled, train_fold['overall_yield'].values)
        oof_mlp[val_idx] = np.clip(mlp.predict(X_va_scaled), 0.0, 100.0)
        
        hgb = HistGradientBoostingRegressor(random_state=seed, max_depth=3, learning_rate=0.05, max_iter=100)
        hgb.fit(X_tr_scaled, train_fold['overall_yield'].values)
        oof_hgb[val_idx] = np.clip(hgb.predict(X_va_scaled), 0.0, 100.0)
        
    return oof_mlp, oof_hgb, y

print("==================================================")
print("TEST 1: Repeated K-Fold CV & Multiple Seeds (Robustness)")
print("==================================================")
seeds = [42, 101, 2023, 7, 888]
n_splits_list = [5, 10]

for n_splits in n_splits_list:
    print(f"\nEvaluating {n_splits}-fold CV across {len(seeds)} different seeds:")
    rmses_blend = []
    for seed in seeds:
        oof_mlp, oof_hgb, y = run_evaluation(df_150, remove_outliers=True, seed=seed, n_splits=n_splits)
        blend = 0.7 * oof_mlp + 0.3 * oof_hgb
        rmse = np.sqrt(mean_squared_error(y, blend))
        rmses_blend.append(rmse)
        print(f"  Seed {seed} RMSE (on full 150 rows): {rmse:.4f}")
    print(f"Mean {n_splits}-fold RMSE across seeds: {np.mean(rmses_blend):.4f} +/- {np.std(rmses_blend):.4f}")

print("\n==================================================")
print("TEST 2: Outlier Cleaning Decision Validation")
print("Evaluating model performance on the FULL 150 rows validation")
print("==================================================")
oof_mlp_clean, oof_hgb_clean, y = run_evaluation(df_150, remove_outliers=True, seed=42, n_splits=5)
blend_clean = 0.7 * oof_mlp_clean + 0.3 * oof_hgb_clean
rmse_clean_on_150 = np.sqrt(mean_squared_error(y, blend_clean))

oof_mlp_full, oof_hgb_full, y = run_evaluation(df_150, remove_outliers=False, seed=42, n_splits=5)
blend_full = 0.7 * oof_mlp_full + 0.3 * oof_hgb_full
rmse_full_on_150 = np.sqrt(mean_squared_error(y, blend_full))

print(f"Trained on 150 rows (including outliers) -> CV RMSE on full 150: {rmse_full_on_150:.4f}")
print(f"Trained on 140 rows (excluding outliers) -> CV RMSE on full 150: {rmse_clean_on_150:.4f}")
print(f"Difference: {rmse_full_on_150 - rmse_clean_on_150:.4f}")

print("\n==================================================")
print("TEST 5: Optimize Ensemble Blending Weights")
print("Testing MLP / HGB weight combinations (on Clean Data training, evaluated on full 150 rows)")
print("==================================================")
best_w = 0.7
min_rmse = 999.0
best_results = {}

for w_mlp in np.linspace(0.0, 1.0, 11):
    w_hgb = 1.0 - w_mlp
    blend_w = w_mlp * oof_mlp_clean + w_hgb * oof_hgb_clean
    rmse_w = np.sqrt(mean_squared_error(y, blend_w))
    print(f"  Weight MLP={w_mlp:.1f}, HGB={w_hgb:.1f} -> RMSE: {rmse_w:.4f}")
    if rmse_w < min_rmse:
        min_rmse = rmse_w
        best_w = w_mlp

print(f"Optimal MLP weight: {best_w:.2f} (RMSE: {min_rmse:.4f})")

best_blend = best_w * oof_mlp_clean + (1.0 - best_w) * oof_hgb_clean

print("\n==================================================")
print("TEST 4: Advanced Performance Metrics")
print(f"Evaluating optimal blend (MLP weight = {best_w:.1f})")
print("==================================================")
rmse = np.sqrt(mean_squared_error(y, best_blend))
mae = mean_absolute_error(y, best_blend)
r2 = r2_score(y, best_blend)
max_err = np.max(np.abs(y - best_blend))

print(f"RMSE:                   {rmse:.4f}")
print(f"MAE:                    {mae:.4f}")
print(f"R² Score:               {r2:.4f}")
print(f"Max Absolute Error:     {max_err:.4f}")

print("\n==================================================")
print("TEST 3: Residual Analysis Across Boundary Conditions")
print("Checking where the prediction errors are coming from")
print("==================================================")
df_analysis = df_150.copy()
df_analysis['pred'] = best_blend
df_analysis['err'] = y - best_blend
df_analysis['abs_err'] = np.abs(df_analysis['err'])
df_analysis['residence_time'] = df_analysis['length_m'] / df_analysis['flow_rate_L_min']

clean_rows = df_analysis.drop(index=outliers_idx)
print("\nTop 5 largest errors in the CLEAN dataset (legitimate points):")
print(clean_rows.sort_values(by='abs_err', ascending=False)[['flow_rate_L_min', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K', 'residence_time', 'overall_yield', 'pred', 'err']].head(5))

print("\nError analysis by feature quantiles:")
for col in ['inlet_temperature_K', 'jacket_temperature_K', 'residence_time', 'concentration_mol_L']:
    low_mask = clean_rows[col] <= clean_rows[col].quantile(0.25)
    high_mask = clean_rows[col] >= clean_rows[col].quantile(0.75)
    mid_mask = ~low_mask & ~high_mask
    
    print(f"  {col}:")
    print(f"    Low (<= 25%): RMSE = {np.sqrt(np.mean(clean_rows.loc[low_mask, 'err']**2)):.4f} (count: {sum(low_mask)})")
    print(f"    Mid (25%-75%): RMSE = {np.sqrt(np.mean(clean_rows.loc[mid_mask, 'err']**2)):.4f} (count: {sum(mid_mask)})")
    print(f"    High (>= 75%): RMSE = {np.sqrt(np.mean(clean_rows.loc[high_mask, 'err']**2)):.4f} (count: {sum(high_mask)})")
