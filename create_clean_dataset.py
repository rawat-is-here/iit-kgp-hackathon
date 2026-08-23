import pandas as pd

df = pd.read_csv('train_dataset.csv')
print("Original shape:", df.shape)

outlier_indices = [97, 77, 149, 86, 30, 23, 142, 18, 57, 26]

df_clean = df.drop(index=outlier_indices).reset_index(drop=True)
print("Cleaned shape:", df_clean.shape)

df_clean.to_csv('trainer_new.csv', index=False)
print("Saved cleaned dataset to trainer_new.csv successfully!")
