import pandas as pd

DATA_PATH = "Data/ecommerce_reviews.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["is_fake_review"].value_counts())

print("\nTarget percentage:")
print(df["is_fake_review"].value_counts(normalize=True) * 100)

print("\nData types:")
print(df.dtypes)

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)