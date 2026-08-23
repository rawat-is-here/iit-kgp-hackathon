import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

df = pd.read_csv('trainer_new.csv')

F = df['flow_rate_L_min'].values
C0 = df['concentration_mol_L'].values
T_in = df['inlet_temperature_K'].values
L = df['length_m'].values
T_j = df['jacket_temperature_K'].values
y_true = df['overall_yield'].values

tau = L / F


def predict_yield(params, F, T_in, L, T_j):
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
    pred = np.clip(pred, 0.0, 100.0)
    return pred

def objective(params):
    pred = predict_yield(params, F, T_in, L, T_j)
    return np.mean((y_true - pred) ** 2)

initial_guess = [
    0.0,
    15.0,
    5000.0,
    18.0,
    7000.0,
    0.0
]

res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[
    (-5.0, 5.0),
    (5.0, 30.0),
    (1000.0, 15000.0),
    (5.0, 35.0),
    (1000.0, 18000.0),
    (-1.0, 1.0)
])

print("Optimization Success:", res.success)
print("Final MSE:", res.fun)
print("Final RMSE:", np.sqrt(res.fun))

best_params = res.x
log_alpha, log_A1, E1, log_A2, E2, log_scale = best_params
print(f"\nFitted parameters:")
print(f"  alpha (heat transfer): {np.exp(log_alpha):.4f}")
print(f"  A1 (freq factor 1):    {np.exp(log_A1):.2e}")
print(f"  E1 (activation energy 1 in K): {E1:.1f}")
print(f"  A2 (freq factor 2):    {np.exp(log_A2):.2e}")
print(f"  E2 (activation energy 2 in K): {E2:.1f}")
print(f"  scale factor:          {np.exp(log_scale):.4f}")

y_pred = predict_yield(best_params, F, T_in, L, T_j)
df['y_phys_pred'] = y_pred

corr = np.corrcoef(y_true, y_pred)[0, 1]
print(f"\nCorrelation coefficient between physical prediction and true yield: {corr:.4f}")
