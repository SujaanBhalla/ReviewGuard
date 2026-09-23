import pandas as pd

DATA_PATH = "data/reviews.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "is_fake_review"

print("=" * 60)
print("FEATURE ANALYSIS")
print("=" * 60)

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\n\nNumeric feature correlations with target:")

numeric_df = df.select_dtypes(include=["number"])

correlations = numeric_df.corr()[TARGET].sort_values(
    ascending=False
)

print(correlations)

print("\n\nCategorical feature analysis:")

categorical_features = [
    "sentiment",
    "verified_purchase",
    "has_title",
    "is_top_reviewer",
    "is_early_review",
    "category",
    "price_tier",
    "brand_tier"
]

for column in categorical_features:

    print("\n" + "-" * 60)
    print(column)

    result = pd.crosstab(
        df[column],
        df[TARGET],
        normalize="index"
    ) * 100

    print(result)