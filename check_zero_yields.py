import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load datasets
df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')
df_test = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/test_dataset.csv')

# Features
features = ['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K']
X_train = df_train[features].copy()
y_train_zero = (df_train['overall_yield'] == 0.0).astype(int)

X_test = df_test[features].copy()

# Add engineered features to both
for df in [X_train, X_test]:
    df['residence_time'] = df['length_m'] / df['flow_rate_L_min']
    df['temp_diff'] = df['jacket_temperature_K'] - df['inlet_temperature_K']
    df['inv_temp_inlet'] = 1.0 / df['inlet_temperature_K']
    df['inv_temp_jacket'] = 1.0 / df['jacket_temperature_K']

# All features list
all_features = list(X_train.columns)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[all_features])
X_test_scaled = scaler.transform(X_test[all_features])

# Train a classifier
clf = RandomForestClassifier(random_state=42, max_depth=4, n_estimators=100)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_train_scaled, y_train_zero, cv=cv, scoring='accuracy')

print(f"Classifier CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

# Fit on all training data
clf.fit(X_train_scaled, y_train_zero)

# Predict on test data
test_preds = clf.predict(X_test_scaled)
test_probs = clf.predict_proba(X_test_scaled)[:, 1]

print(f"\nPredicted zero-yield count in test dataset: {sum(test_preds)}")
print("Probabilities of zero-yield for the test rows:")
for idx, prob in enumerate(test_probs):
    if prob > 0.3:
        print(f"Row {idx+1}: prob={prob:.4f}")
