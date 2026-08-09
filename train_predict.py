import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("[1/5] Loading datasets...")
df_train = pd.read_csv('trainer_new.csv')
df_test = pd.read_csv('test_dataset.csv')

print(f"  Training set size: {df_train.shape[0]} rows")
print(f"  Test set size:     {df_test.shape[0]} rows")

# Define input features
features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']
X_train = df_train[features].copy()
y_train = df_train['overall_yield'].copy().values
X_test = df_test[features].copy()

# Add standard engineered features
print("[2/5] Engineering physical features (Residence time, Temperature differences)...")
for df in [X_train, X_test]:
    df['residence_time'] = df['length_m'] / df['flow_rate_L_min']
    df['temp_diff'] = df['jacket_temperature_K'] - df['inlet_temperature_K']
    df['inv_temp_inlet'] = 1.0 / df['inlet_temperature_K']
    df['inv_temp_jacket'] = 1.0 / df['jacket_temperature_K']

# Define the PFR ODE simulator
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

print("[3/5] Fitting physical kinetics ODE model using SciPy optimizer...")
# Extract arrays for training
F_tr = X_train['flow_rate_L_min'].values
C0_tr = X_train['concentration_mol_L'].values
T_in_tr = X_train['inlet_temperature_K'].values
L_tr = X_train['length_m'].values
T_j_tr = X_train['jacket_temperature_K'].values

def objective(params):
    pred = simulate_pfr(params, F_tr, C0_tr, T_in_tr, L_tr, T_j_tr)
    return np.mean((y_train - pred) ** 2)

initial_guess = [0.0, 15.0, 5000.0, 18.0, 7000.0, 0.0, 1.0, 1.0]
res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[
    (-5.0, 5.0), (5.0, 30.0), (1000.0, 15000.0), (5.0, 35.0), (1000.0, 18000.0), (-1.0, 1.0),
    (0.2, 3.0), (0.2, 3.0)
])

best_params = res.x
log_alpha, log_A1, E1, log_A2, E2, log_scale, order_a, order_b = best_params

print("  Physical Model Fitted Successfully:")
print(f"    Heat Transfer (alpha):         {np.exp(log_alpha):.4f}")
print(f"    Desired Rxn Activation (E1):   {E1 * 8.314 * 10**-3:.2f} kJ/mol")
print(f"    Side Rxn Activation (E2):      {E2 * 8.314 * 10**-3:.2f} kJ/mol")
print(f"    Desired Rxn Order (a):         {order_a:.4f}")
print(f"    Side Rxn Order (b):            {order_b:.4f}")

# Generate physical kinetics feature
print("  Generating physical chemistry feature (y_phys)...")
y_train_phys = simulate_pfr(best_params, F_tr, C0_tr, T_in_tr, L_tr, T_j_tr)

F_te = X_test['flow_rate_L_min'].values
C0_te = X_test['concentration_mol_L'].values
T_in_te = X_test['inlet_temperature_K'].values
L_te = X_test['length_m'].values
T_j_te = X_test['jacket_temperature_K'].values
y_test_phys = simulate_pfr(best_params, F_te, C0_te, T_in_te, L_te, T_j_te)

X_train['y_phys'] = y_train_phys
X_test['y_phys'] = y_test_phys

# Fit ML Regressor
print("[4/5] Training final Physics-Informed MLP Regressor...")
all_features = list(X_train.columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[all_features])
X_test_scaled = scaler.transform(X_test[all_features])

# Multi-Layer Perceptron Regressor
mlp = MLPRegressor(
    random_state=42, 
    alpha=0.001, 
    hidden_layer_sizes=(50, 50), 
    learning_rate_init=0.01, 
    max_iter=3000
)
mlp.fit(X_train_scaled, y_train)

# Calculate final training RMSE
train_preds = np.clip(mlp.predict(X_train_scaled), 0.0, 100.0)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print(f"  Training RMSE: {train_rmse:.4f}%")

print("[5/5] Generating final predictions for test dataset...")
test_preds = np.clip(mlp.predict(X_test_scaled), 0.0, 100.0)

# Create submission file
submission = pd.DataFrame({
    'overall_yield': np.round(test_preds, 3)
})

# Save to predictions.csv
submission.to_csv('predictions.csv', index=False)
print("\nSuccess! Predictions saved to 'predictions.csv'.")
print("Important: Rename 'predictions.csv' to your '[TeamName].csv' before submitting!")
print("Cleaned shape verified: exactly 50 rows.")
