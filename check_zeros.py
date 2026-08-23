import pandas as pd

df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')
df_train['residence_time'] = df_train['length_m'] / df_train['flow_rate_L_min']

df_zero = df_train[df_train['overall_yield'] == 0.0]

print("Zero-yield rows:")
pd.set_option('display.max_rows', None)
print(df_zero[['flow_rate_L_min', 'concentration_mol_L', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K', 'residence_time']])
