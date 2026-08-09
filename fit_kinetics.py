import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# Load cleaned dataset
df = pd.read_csv('trainer_new.csv')

# Inputs
F = df['flow_rate_L_min'].values       # Flow rate L/min
C0 = df['concentration_mol_L'].values   # Inlet concentration mol/L
T_in = df['inlet_temperature_K'].values # Inlet Temp K
L = df['length_m'].values               # Length m
T_j = df['jacket_temperature_K'].values # Jacket Temp K
y_true = df['overall_yield'].values     # Target Yield %

# Residence time tau = L / F
tau = L / F

# We want to fit:
# T_eff = T_j - (T_j - T_in) * exp(-alpha * tau)
# k1 = A1 * exp(-E1 / T_eff)
# k2 = A2 * exp(-E2 / T_eff)
# Yield = scale * (k1 / (k2 - k1)) * (exp(-k1 * tau) - exp(-k2 * tau)) * 100

def predict_yield(params, F, T_in, L, T_j):
    # Unpack parameters
    # We use log-parameters for scale factors to ensure they stay positive
    log_alpha, log_A1, E1, log_A2, E2, log_scale = params
    
    alpha = np.exp(log_alpha)
    A1 = np.exp(log_A1)
    A2 = np.exp(log_A2)
    scale = np.exp(log_scale)
    
    # Calculate effective temperature along the reactor
    # Since heat transfer takes time, let's assume average T_eff
    # exp(-alpha * tau) determines how close T_eff gets to T_jacket
    tau = L / F
    T_eff = T_j - (T_j - T_in) * np.exp(-alpha * tau)
    
    # Clip T_eff to avoid division by zero or extreme values
    T_eff = np.clip(T_eff, 100.0, 1000.0)
    
    # Rate constants
    # Divide by 1000 to keep exponents in reasonable range
    k1 = A1 * np.exp(-E1 / T_eff)
    k2 = A2 * np.exp(-E2 / T_eff)
    
    # Avoid division by zero
    diff = k2 - k1
    diff = np.where(np.abs(diff) < 1e-5, 1e-5 * np.sign(diff), diff)
    
    # Analytical yield formula
    pred = scale * (k1 / diff) * (np.exp(-k1 * tau) - np.exp(-k2 * tau)) * 100.0
    # Yield cannot be negative or exceed 100% (or scale * 100)
    pred = np.clip(pred, 0.0, 100.0)
    return pred

def objective(params):
    pred = predict_yield(params, F, T_in, L, T_j)
    return np.mean((y_true - pred) ** 2)

# Initial guess
# log_alpha, log_A1, E1, log_A2, E2, log_scale
# E1 and E2 are activation energies in Kelvin. E/R ~ 5000 to 12000.
# Let's assume A1 and A2 are large, so log(A) around 10 to 20
initial_guess = [
    0.0,    # log(alpha) = 0 -> alpha = 1.0
    15.0,   # log(A1) = 15
    5000.0, # E1 = 5000
    18.0,   # log(A2) = 18 (side reaction has higher frequency factor)
    7000.0, # E2 = 7000 (side reaction has higher activation energy)
    0.0     # log(scale) = 0 -> scale = 1.0
]

# Run optimization
res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[
    (-5.0, 5.0),     # log_alpha
    (5.0, 30.0),     # log_A1
    (1000.0, 15000.0),# E1
    (5.0, 35.0),     # log_A2
    (1000.0, 18000.0),# E2
    (-1.0, 1.0)      # log_scale
])

print("Optimization Success:", res.success)
print("Final MSE:", res.fun)
print("Final RMSE:", np.sqrt(res.fun))

# Best parameters
best_params = res.x
log_alpha, log_A1, E1, log_A2, E2, log_scale = best_params
print(f"\nFitted parameters:")
print(f"  alpha (heat transfer): {np.exp(log_alpha):.4f}")
print(f"  A1 (freq factor 1):    {np.exp(log_A1):.2e}")
print(f"  E1 (activation energy 1 in K): {E1:.1f}")
print(f"  A2 (freq factor 2):    {np.exp(log_A2):.2e}")
print(f"  E2 (activation energy 2 in K): {E2:.1f}")
print(f"  scale factor:          {np.exp(log_scale):.4f}")

# Check predictions
y_pred = predict_yield(best_params, F, T_in, L, T_j)
df['y_phys_pred'] = y_pred

# Print correlation between physical prediction and true yield
corr = np.corrcoef(y_true, y_pred)[0, 1]
print(f"\nCorrelation coefficient between physical prediction and true yield: {corr:.4f}")
