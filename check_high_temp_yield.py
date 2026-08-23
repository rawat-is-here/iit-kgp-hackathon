import pandas as pd

df_train = pd.read_csv(r'c:/Users/akash/OneDrive/Desktop/IIT KGP/train_dataset.csv')
df_train['residence_time'] = df_train['length_m'] / df_train['flow_rate_L_min']

df_hot_slow = df_train[(df_train['jacket_temperature_K'] > 480.0) & 
                       (df_train['inlet_temperature_K'] > 450.0) & 
                       (df_train['residence_time'] > 0.8)]

print("Hot and slow rows (should have low yield):")
print(df_hot_slow[['flow_rate_L_min', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K', 'residence_time', 'overall_yield']])

df_high_yield = df_train[df_train['overall_yield'] > 80.0]
print("\nSome high yield rows:")
print(df_high_yield[['flow_rate_L_min', 'inlet_temperature_K', 'length_m', 'jacket_temperature_K', 'residence_time', 'overall_yield']].head(10))
