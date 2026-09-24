import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ReviewGuard",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# LOAD MODEL + DATA
# ============================================================

MODEL_PATH = "models/random_forest_model.pkl"
DATA_PATH = "Data/ecommerce_reviews.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ============================================================
# HEADER
# ============================================================

st.title("🛡️ ReviewGuard")
st.subheader("AI Review Trust & Risk Analyzer")

st.write(
    "Analyze individual reviews and explore overall product review health "
    "using machine learning and behavioral signals."
)

st.divider()

# ============================================================
# SECTION 1 — SINGLE REVIEW ANALYZER
# ============================================================

st.header("🔍 Single Review Analyzer")

st.info(
    "Enter the review characteristics below and click **Analyze Review** "
    "to generate a prediction."
)

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input(
        "Year",
        min_value=2020,
        max_value=2030,
        value=2026
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=9
    )

    star_rating = st.slider(
        "Star Rating",
        min_value=1,
        max_value=5,
        value=5
    )

    sentiment = st.selectbox(
        "Sentiment",
        ["positive", "neutral", "negative"]
    )

    verified_purchase = st.selectbox(
        "Verified Purchase",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    review_length_words = st.number_input(
        "Review Length (words)",
        min_value=1,
        max_value=1000,
        value=25
    )

with col2:
    has_title = st.selectbox(
        "Has Title",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    title_word_count = st.number_input(
        "Title Word Count",
        min_value=0,
        max_value=100,
        value=4
    )

    num_images_attached = st.number_input(
        "Images Attached",
        min_value=0,
        max_value=50,
        value=0
    )

    helpful_votes = st.number_input(
        "Helpful Votes",
        min_value=0,
        max_value=10000,
        value=0
    )

    total_votes = st.number_input(
        "Total Votes",
        min_value=0,
        max_value=10000,
        value=1
    )

    if total_votes > 0:
        helpful_ratio = helpful_votes / total_votes
    else:
        helpful_ratio = 0.0

with col3:
    days_since_purchase = st.number_input(
        "Days Since Purchase",
        min_value=0,
        max_value=1000,
        value=2
    )

    reviewer_review_count = st.number_input(
        "Reviewer Review Count",
        min_value=0,
        max_value=1000,
        value=1
    )

    is_top_reviewer = st.selectbox(
        "Top Reviewer",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_early_review = st.selectbox(
        "Early Review",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    exclamation_marks = st.number_input(
        "Exclamation Marks",
        min_value=0,
        max_value=100,
        value=0
    )

    all_caps_ratio = st.number_input(
        "ALL CAPS Ratio",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    readability_score = st.number_input(
        "Readability Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

# ============================================================
# OTHER FEATURES
# ============================================================

col4, col5, col6 = st.columns(3)

with col4:
    category = st.selectbox(
        "Category",
        sorted(df["category"].dropna().unique().tolist())
    )

with col5:
    price_usd = st.number_input(
        "Price (USD)",
        min_value=0.0,
        max_value=100000.0,
        value=49.99
    )

with col6:
    price_tier = st.selectbox(
        "Price Tier",
        sorted(df["price_tier"].dropna().unique().tolist())
    )

brand_tier = st.selectbox(
    "Brand Tier",
    sorted(df["brand_tier"].dropna().unique().tolist())
)

st.caption(f"Calculated Helpful Ratio: {helpful_ratio:.2f}")

# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Review",
    type="primary",
    use_container_width=True
)

if analyze_button:

    input_data = pd.DataFrame([{
        "year": year,
        "month": month,
        "star_rating": star_rating,
        "sentiment": sentiment,
        "verified_purchase": verified_purchase,
        "review_length_words": review_length_words,
        "has_title": has_title,
        "title_word_count": title_word_count,
        "num_images_attached": num_images_attached,
        "helpful_votes": helpful_votes,
        "total_votes": total_votes,
        "helpful_ratio": helpful_ratio,
        "days_since_purchase": days_since_purchase,
        "reviewer_review_count": reviewer_review_count,
        "is_top_reviewer": is_top_reviewer,
        "is_early_review": is_early_review,
        "exclamation_marks": exclamation_marks,
        "all_caps_ratio": all_caps_ratio,
        "readability_score": readability_score,
        "category": category,
        "price_usd": price_usd,
        "price_tier": price_tier,
        "brand_tier": brand_tier
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    fake_probability = probabilities[1] * 100
    genuine_probability = probabilities[0] * 100

    trust_score = round(genuine_probability, 1)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()
    st.header("📋 Analysis Result")

    if prediction == 1:
        st.error("⚠️ SUSPICIOUS / POTENTIALLY FAKE")
    else:
        st.success("✅ LIKELY GENUINE")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "🛡️ Trust Score",
            f"{trust_score}/100"
        )

    with result_col2:
        st.metric(
            "⚠️ Model Risk",
            f"{fake_probability:.1f}%"
        )

    with result_col3:
        st.metric(
            "✅ Genuine Signal",
            f"{genuine_probability:.1f}%"
        )

    st.progress(int(trust_score))

    if trust_score >= 70:
        st.success("Relatively Trustworthy")
    elif trust_score >= 40:
        st.warning("Needs Attention")
    else:
        st.error("High Suspicion")

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.subheader("🧠 Why did the model flag this review?")

    reasons = []

    if all_caps_ratio >= 10:
        reasons.append("High use of capital letters")

    if exclamation_marks >= 3:
        reasons.append("Excessive exclamation marks")

    if star_rating == 5:
        reasons.append("Maximum 5-star rating")

    if review_length_words <= 10:
        reasons.append("Very short review")

    if days_since_purchase <= 3:
        reasons.append("Review posted shortly after purchase")

    if verified_purchase == 0:
        reasons.append("Purchase is not verified")

    if reviewer_review_count <= 2:
        reasons.append("Reviewer has limited review history")

    if sentiment == "positive":
        reasons.append("Strong positive sentiment")

    if reasons:
        for reason in reasons:
            st.write(f"• {reason}")
    else:
        st.write("No major behavioral warning signals detected.")

    st.caption(
        "⚠️ Trust Score is a risk indicator generated by the model. "
        "It does not prove that a review is fake or genuine."
    )

# ============================================================
# SECTION 2 — PRODUCT REVIEW HEALTH DASHBOARD
# ============================================================

st.divider()

st.header("📊 Product Review Health Dashboard")

st.write(
    "Explore the overall fake-review pattern in the dataset."
)

# ============================================================
# DATASET METRICS
# ============================================================

total_reviews = len(df)

fake_reviews = int(
    (df["is_fake_review"] == 1).sum()
)

genuine_reviews = int(
    (df["is_fake_review"] == 0).sum()
)

fake_percentage = (
    fake_reviews / total_reviews * 100
    if total_reviews > 0 else 0
)

overall_trust = 100 - fake_percentage

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "📝 Total Reviews",
        f"{total_reviews:,}"
    )

with metric2:
    st.metric(
        "⚠️ Suspicious Reviews",
        f"{fake_reviews:,}"
    )

with metric3:
    st.metric(
        "✅ Genuine Reviews",
        f"{genuine_reviews:,}"
    )

with metric4:
    st.metric(
        "🛡️ Overall Trust",
        f"{overall_trust:.1f}%"
    )

# ============================================================
# GENUINE VS SUSPICIOUS
# ============================================================

st.subheader("🔎 Genuine vs Suspicious Reviews")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    review_counts = pd.DataFrame({
        "Review Type": ["Genuine", "Suspicious"],
        "Count": [genuine_reviews, fake_reviews]
    })

    st.bar_chart(
        review_counts.set_index("Review Type")
    )

with chart_col2:

    review_percentage = pd.DataFrame({
        "Review Type": ["Genuine", "Suspicious"],
        "Percentage": [
            genuine_reviews / total_reviews * 100,
            fake_reviews / total_reviews * 100
        ]
    })

    st.bar_chart(
        review_percentage.set_index("Review Type")
    )

# ============================================================
# STAR RATING ANALYSIS
# ============================================================

st.subheader("⭐ Suspicious Reviews by Star Rating")

star_analysis = (
    df.groupby("star_rating")["is_fake_review"]
    .mean()
    .mul(100)
    .reset_index()
)

star_analysis.columns = [
    "Star Rating",
    "Suspicious Percentage"
]

st.bar_chart(
    star_analysis.set_index("Star Rating")
)

# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

st.subheader("😊 Suspicious Reviews by Sentiment")

sentiment_analysis = (
    df.groupby("sentiment")["is_fake_review"]
    .mean()
    .mul(100)
    .reset_index()
)

sentiment_analysis.columns = [
    "Sentiment",
    "Suspicious Percentage"
]

st.bar_chart(
    sentiment_analysis.set_index("Sentiment")
)

# ============================================================
# MONTHLY TREND
# ============================================================

st.subheader("📈 Suspicious Review Trend")

monthly_analysis = (
    df.groupby(["year", "month"])["is_fake_review"]
    .mean()
    .mul(100)
    .reset_index()
)

monthly_analysis["Period"] = (
    monthly_analysis["year"].astype(str)
    + "-"
    + monthly_analysis["month"].astype(str).str.zfill(2)
)

monthly_analysis = monthly_analysis.sort_values(
    ["year", "month"]
)

trend_data = monthly_analysis[
    ["Period", "is_fake_review"]
].set_index("Period")

trend_data.columns = ["Suspicious Percentage"]

st.line_chart(trend_data)

# ============================================================
# CATEGORY ANALYSIS
# ============================================================

st.subheader("🛍️ Suspicious Reviews by Product Category")

category_analysis = (
    df.groupby("category")["is_fake_review"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)

category_analysis.columns = [
    "Category",
    "Suspicious Percentage"
]

st.bar_chart(
    category_analysis.set_index("Category")
)

# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Model Information")

st.write(
    "**Model:** Random Forest Classifier"
)

st.write(
    "**Purpose:** Detect suspicious review behavior using review, "
    "reviewer, purchase, sentiment and behavioral features."
)

st.write(
    "**Output:** Suspicious/Genuine prediction + risk probability + "
    "explainability signals."
)

st.caption(
    "Dataset contains synthetic e-commerce review data. "
    "Model performance should therefore be interpreted within the "
    "context of this dataset."
)