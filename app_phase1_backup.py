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
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

# ============================================================
# HEADER
# ============================================================

st.title("🛡️ ReviewGuard")
st.subheader("AI-Powered Review Trust & Risk Analyzer")

st.write(
    "Analyze review behavior and identify patterns associated "
    "with potentially suspicious e-commerce reviews."
)

st.divider()

# ============================================================
# REVIEW INFORMATION
# ============================================================

st.header("🔍 Review Information")

col1, col2, col3 = st.columns(3)

with col1:

    star_rating = st.slider(
        "⭐ Star Rating",
        1,
        5,
        5
    )

    verified_purchase = st.selectbox(
        "🛒 Verified Purchase",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    review_length_words = st.number_input(
        "📝 Review Length (words)",
        min_value=1,
        max_value=1000,
        value=25
    )

    has_title = st.selectbox(
        "📌 Has Title?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    title_word_count = st.number_input(
        "Title Word Count",
        min_value=0,
        max_value=50,
        value=4
    )

with col2:

    exclamation_marks = st.number_input(
        "❗ Exclamation Marks",
        min_value=0,
        max_value=50,
        value=1
    )

    all_caps_ratio = st.slider(
        "🔠 CAPS Ratio (%)",
        0.0,
        100.0,
        5.0
    )

    num_images_attached = st.number_input(
        "🖼️ Images Attached",
        min_value=0,
        max_value=20,
        value=0
    )

    days_since_purchase = st.number_input(
        "📅 Days Since Purchase",
        min_value=0,
        max_value=1000,
        value=30
    )

    reviewer_review_count = st.number_input(
        "👤 Previous Reviews by Reviewer",
        min_value=0,
        max_value=1000,
        value=10
    )

with col3:

    is_top_reviewer = st.selectbox(
        "🏆 Top Reviewer?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_early_review = st.selectbox(
        "⚡ Early Review?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    sentiment = st.selectbox(
        "😊 Sentiment",
        ["positive", "neutral", "negative"]
    )

    category = st.selectbox(
        "📦 Product Category",
        [
            "Automotive",
            "Beauty",
            "Books",
            "Clothing",
            "Electronics",
            "Food & Grocery",
            "Health",
            "Home & Kitchen",
            "Sports & Outdoors",
            "Toys & Games"
        ]
    )

    price_usd = st.number_input(
        "💰 Product Price (USD)",
        min_value=0.0,
        max_value=10000.0,
        value=49.99
    )

# ============================================================
# ADDITIONAL FEATURES
# ============================================================

st.divider()

st.header("📊 Additional Review Signals")

col4, col5, col6 = st.columns(3)

with col4:

    year = st.number_input(
        "Year",
        2020,
        2030,
        2026
    )

    month = st.number_input(
        "Month",
        1,
        12,
        9
    )

    helpful_votes = st.number_input(
        "Helpful Votes",
        min_value=0,
        max_value=10000,
        value=0
    )

with col5:

    total_votes = st.number_input(
        "Total Votes",
        min_value=0,
        max_value=10000,
        value=1
    )

    readability_score = st.number_input(
        "Readability Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

with col6:

    price_tier = st.selectbox(
        "Price Tier",
        [
            "budget",
            "low",
            "mid",
            "high",
            "premium"
        ]
    )

    brand_tier = st.selectbox(
        "Brand Tier",
        [
            "budget",
            "mid",
            "premium"
        ]
    )

# ============================================================
# HELPFUL RATIO
# ============================================================

if total_votes > 0:
    helpful_ratio = helpful_votes / total_votes
else:
    helpful_ratio = 0.0

# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze_button = st.button(
    "🔍 Analyze Review",
    type="primary",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if analyze_button:

    review = pd.DataFrame([{

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

    prediction = model.predict(review)[0]

    probabilities = model.predict_proba(review)[0]

    genuine_probability = probabilities[0] * 100
    fake_probability = probabilities[1] * 100

    # ========================================================
    # TRUST SCORE
    # ========================================================

    trust_score = round(100 - fake_probability, 1)

    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.divider()

    st.header("🛡️ ReviewGuard Analysis")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.error("⚠️ SUSPICIOUS / POTENTIALLY FAKE")
        else:
            st.success("✅ LIKELY GENUINE")

    with result_col2:

        st.metric(
            "🎯 Trust Score",
            f"{trust_score}/100"
        )

    # ========================================================
    # TRUST SCORE INTERPRETATION
    # ========================================================

    if trust_score >= 70:

        st.success(
            f"🟢 Relatively Trustworthy — {trust_score}/100"
        )

    elif trust_score >= 40:

        st.warning(
            f"🟠 Needs Attention — {trust_score}/100"
        )

    else:

        st.error(
            f"🔴 High Suspicion — {trust_score}/100"
        )

    # ========================================================
    # RISK METRICS
    # ========================================================

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "🛡️ Trust Score",
            f"{trust_score:.1f}/100"
        )

    with metric2:

        st.metric(
            "⚠️ Model Risk",
            f"{fake_probability:.1f}%"
        )

    with metric3:

        st.metric(
            "✅ Genuine Signal",
            f"{genuine_probability:.1f}%"
        )

    # ========================================================
    # RISK BAR
    # ========================================================

    st.write("### 📊 Suspicious Review Risk")

    st.progress(
        min(int(fake_probability), 100)
    )

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.divider()

    st.header("🧠 Why did ReviewGuard flag this review?")

    signals = []

    if all_caps_ratio >= 10:

        signals.append(
            "🔴 High capitalization usage"
        )

    if exclamation_marks >= 3:

        signals.append(
            "🔴 Excessive exclamation marks"
        )

    if star_rating == 5:

        signals.append(
            "🟠 Maximum 5-star rating"
        )

    if review_length_words <= 10:

        signals.append(
            "🟠 Very short review"
        )

    if days_since_purchase <= 3:

        signals.append(
            "🟠 Review posted very soon after purchase"
        )

    if verified_purchase == 0:

        signals.append(
            "🟠 Purchase is not verified"
        )

    if reviewer_review_count <= 2:

        signals.append(
            "🟠 Limited reviewer history"
        )

    if sentiment == "positive":

        signals.append(
            "🟡 Positive sentiment"
        )

    if len(signals) == 0:

        signals.append(
            "🟢 No major behavioral warning signals detected."
        )

    for signal in signals:

        st.write(signal)

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.divider()

    with st.expander("🤖 Model Information"):

        st.write(
            "ReviewGuard currently uses a Random Forest "
            "classification model trained on behavioral review features."
        )

        st.write(
            "The model estimates the likelihood that the review "
            "belongs to the suspicious class in the training dataset."
        )

        st.write(
            "The Trust Score is calculated as:"
        )

        st.code(
            "Trust Score = 100 − Suspicious Risk"
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ ReviewGuard provides a risk-based prediction using "
        "behavioral signals. A suspicious prediction does not prove "
        "that a review is actually fake."
    )