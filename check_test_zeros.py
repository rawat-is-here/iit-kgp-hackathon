import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')
df_test = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/test_dataset.csv')

features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']
X_train = df_train[features].copy()
y_train_zero = (df_train['overall_yield'] == 0.0).astype(int)
X_test = df_test[features].copy()

for df in [X_train, X_test]:
    df['residence_time'] = df['length_m'] / df['flow_rate_L_min']
    df['temp_diff'] = df['jacket_temperature_K'] - df['inlet_temperature_K']
    df['inv_temp_inlet'] = 1.0 / df['inlet_temperature_K']
    df['inv_temp_jacket'] = 1.0 / df['jacket_temperature_K']

all_features = list(X_train.columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[all_features])
X_test_scaled = scaler.transform(X_test[all_features])

clf = RandomForestClassifier(random_state=42, max_depth=4, n_estimators=100)
clf.fit(X_train_scaled, y_train_zero)

test_probs = clf.predict_proba(X_test_scaled)[:, 1]
df_test['zero_prob'] = test_probs
df_test['residence_time'] = df_test['length_m'] / df_test['flow_rate_L_min']

print("Test rows with high probability of zero-yield (> 0.5):")
print(df_test[df_test['zero_prob'] > 0.5][['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K', 'residence_time', 'zero_prob']].sort_values(by='zero_prob', ascending=False))
