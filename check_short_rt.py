import pandas as pd

df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')
df_train['residence_time'] = df_train['length_m'] / df_train['flow_rate_L_min']

df_short = df_train[df_train['residence_time'] < 0.3].sort_values(by='residence_time')
print("Rows with short residence times (< 0.3):")
print(df_short[['flow_rate_L_min', 'length_m', 'residence_time', 'inlet_temperature_K', 'jacket_temperature_K', 'overall_yield']])
