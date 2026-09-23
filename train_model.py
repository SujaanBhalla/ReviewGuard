import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "Data/ecommerce_reviews.csv"
TARGET = "is_fake_review"

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Shape:", df.shape)

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

columns_to_drop = [
    "review_id",
    "product_id",
    "user_id",
    "review_date"
]

X = X.drop(columns=columns_to_drop)

numeric_features = [
    "year",
    "month",
    "star_rating",
    "review_length_words",
    "title_word_count",
    "num_images_attached",
    "helpful_votes",
    "total_votes",
    "helpful_ratio",
    "days_since_purchase",
    "reviewer_review_count",
    "exclamation_marks",
    "all_caps_ratio",
    "readability_score",
    "price_usd"
]

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

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# ============================================================
# MODEL 1 — LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

print("\n" + "=" * 60)
print("TRAINING MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

# ============================================================
# MODEL 2 — RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\n" + "=" * 60)
print("TRAINING MODEL 2: RANDOM FOREST")
print("=" * 60)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, y_true, predictions):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Genuine", "Fake"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(confusion_matrix(y_true, predictions))


evaluate_model(
    "MODEL 1 — LOGISTIC REGRESSION",
    y_test,
    logistic_predictions
)

evaluate_model(
    "MODEL 2 — RANDOM FOREST",
    y_test,
    random_forest_predictions
)

# ============================================================
# SAVE MODELS
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    logistic_model,
    "models/logistic_model.pkl"
)

joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)

print("\n" + "=" * 60)
print("MODELS SAVED")
print("=" * 60)

print("Saved:")
print("models/logistic_model.pkl")
print("models/random_forest_model.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)