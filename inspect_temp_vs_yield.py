import pandas as pd
import numpy as np

df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')
df_train['max_temp'] = df_train[['inlet_temperature_K', 'jacket_temperature_K']].max(axis=1)

print("Rows where maximum temperature is low (< 380 K):")
print(df_train[df_train['max_temp'] < 380.0][['flow_rate_L_min', 'inlet_temperature_K', 'jacket_temperature_K', 'max_temp', 'overall_yield']])

print("\nRows where maximum temperature is very low (< 370 K):")
print(df_train[df_train['max_temp'] < 370.0][['flow_rate_L_min', 'inlet_temperature_K', 'jacket_temperature_K', 'max_temp', 'overall_yield']])

print("\nLet's also look at rows where yield is very high (> 80%) but temperature is moderate:")
print(df_train[(df_train['overall_yield'] > 80.0) & (df_train['max_temp'] < 420.0)][['flow_rate_L_min', 'inlet_temperature_K', 'jacket_temperature_K', 'max_temp', 'overall_yield']])
