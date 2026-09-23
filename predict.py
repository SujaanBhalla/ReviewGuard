import joblib
import pandas as pd

# Load trained model
MODEL_PATH = "models/random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

# Sample review information
review = pd.DataFrame([{
    "year": 2026,
    "month": 9,
    "star_rating": 5,
    "sentiment": "positive",
    "verified_purchase": 0,
    "review_length_words": 25,
    "has_title": 1,
    "title_word_count": 4,
    "num_images_attached": 0,
    "helpful_votes": 0,
    "total_votes": 1,
    "helpful_ratio": 0.0,
    "days_since_purchase": 2,
    "reviewer_review_count": 1,
    "is_top_reviewer": 0,
    "is_early_review": 1,
    "exclamation_marks": 6,
    "all_caps_ratio": 25.0,
    "readability_score": 70.0,
    "category": "Electronics",
    "price_usd": 49.99,
    "price_tier": "mid",
    "brand_tier": "premium"
}])

# Make prediction
prediction = model.predict(review)[0]

# Get probability
probability = model.predict_proba(review)[0]

fake_probability = probability[1] * 100
genuine_probability = probability[0] * 100

print("=" * 60)
print("REVIEWGUARD — REVIEW ANALYSIS")
print("=" * 60)

if prediction == 1:
    print("\nPrediction: ⚠️ SUSPICIOUS / FAKE")
else:
    print("\nPrediction: ✅ LIKELY GENUINE")

print(f"Fake Risk      : {fake_probability:.2f}%")
print(f"Genuine Chance : {genuine_probability:.2f}%")

print("\n" + "=" * 60)