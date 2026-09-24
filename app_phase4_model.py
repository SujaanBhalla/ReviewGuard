import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ReviewGuard",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/random_forest_model.pkl"
DATA_PATH = "Data/ecommerce_reviews.csv"

# ============================================================
# LOAD MODEL + DATA
# ============================================================

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ============================================================
# HEADER
# ============================================================

st.title("🛡️ ReviewGuard")
st.subheader("AI Review Trust & Risk Analyzer")

st.write(
    "Analyze individual reviews, explore product review health, "
    "and compare machine learning models for suspicious-review detection."
)

st.divider()

# ============================================================
# SECTION 1 — SINGLE REVIEW ANALYZER
# ============================================================

st.header("🔍 Single Review Analyzer")

st.info(
    "Enter the review characteristics below and click "
    "**Analyze Review** to generate a prediction."
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
        sorted(
            df["category"]
            .dropna()
            .unique()
            .tolist()
        )
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
        sorted(
            df["price_tier"]
            .dropna()
            .unique()
            .tolist()
        )
    )

brand_tier = st.selectbox(
    "Brand Tier",
    sorted(
        df["brand_tier"]
        .dropna()
        .unique()
        .tolist()
    )
)

st.caption(
    f"Calculated Helpful Ratio: {helpful_ratio:.2f}"
)

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

    st.progress(
        int(trust_score)
    )

    if trust_score >= 70:

        st.success(
            "Relatively Trustworthy"
        )

    elif trust_score >= 40:

        st.warning(
            "Needs Attention"
        )

    else:

        st.error(
            "High Suspicion"
        )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "🧠 Why did the model flag this review?"
    )

    reasons = []

    if all_caps_ratio >= 10:
        reasons.append(
            "High use of capital letters"
        )

    if exclamation_marks >= 3:
        reasons.append(
            "Excessive exclamation marks"
        )

    if star_rating == 5:
        reasons.append(
            "Maximum 5-star rating"
        )

    if review_length_words <= 10:
        reasons.append(
            "Very short review"
        )

    if days_since_purchase <= 3:
        reasons.append(
            "Review posted shortly after purchase"
        )

    if verified_purchase == 0:
        reasons.append(
            "Purchase is not verified"
        )

    if reviewer_review_count <= 2:
        reasons.append(
            "Reviewer has limited review history"
        )

    if sentiment == "positive":
        reasons.append(
            "Strong positive sentiment"
        )

    if reasons:

        for reason in reasons:

            st.write(
                f"• {reason}"
            )

    else:

        st.write(
            "No major behavioral warning signals detected."
        )

    st.caption(
        "⚠️ Trust Score is a risk indicator generated by the model. "
        "It does not prove that a review is fake or genuine."
    )

# ============================================================
# SECTION 2 — PRODUCT REVIEW HEALTH
# ============================================================

st.divider()

st.header(
    "🛍️ Product Review Health"
)

st.write(
    "Select a product to analyze the review ecosystem associated "
    "with that product."
)

product_ids = sorted(
    df["product_id"]
    .dropna()
    .unique()
    .tolist()
)

selected_product = st.selectbox(
    "🔎 Select Product ID",
    product_ids
)

product_df = df[
    df["product_id"] == selected_product
].copy()

product_total = len(product_df)

product_fake = int(
    (product_df["is_fake_review"] == 1).sum()
)

product_genuine = int(
    (product_df["is_fake_review"] == 0).sum()
)

product_fake_percentage = (
    product_fake / product_total * 100
    if product_total > 0
    else 0
)

product_genuine_percentage = (
    product_genuine / product_total * 100
    if product_total > 0
    else 0
)

product_trust_score = product_genuine_percentage

if product_trust_score >= 90:

    risk_status = "🟢 Relatively Healthy"

elif product_trust_score >= 70:

    risk_status = "🟡 Needs Attention"

else:

    risk_status = "🔴 High Suspicion"

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.metric(
        "📝 Total Reviews",
        f"{product_total:,}"
    )

with p2:

    st.metric(
        "⚠️ Suspicious",
        f"{product_fake:,}"
    )

with p3:

    st.metric(
        "✅ Genuine",
        f"{product_genuine:,}"
    )

with p4:

    st.metric(
        "🛡️ Product Trust",
        f"{product_trust_score:.1f}%"
    )

st.write(
    f"### Product Status: {risk_status}"
)

st.progress(
    int(product_trust_score)
)

# ============================================================
# PRODUCT REVIEW BREAKDOWN
# ============================================================

st.subheader(
    "📊 Review Breakdown"
)

product_chart_col1, product_chart_col2 = st.columns(2)

with product_chart_col1:

    product_counts = pd.DataFrame({
        "Review Type": [
            "Genuine",
            "Suspicious"
        ],
        "Count": [
            product_genuine,
            product_fake
        ]
    })

    st.bar_chart(
        product_counts.set_index(
            "Review Type"
        )
    )

with product_chart_col2:

    product_percentages = pd.DataFrame({
        "Review Type": [
            "Genuine",
            "Suspicious"
        ],
        "Percentage": [
            product_genuine_percentage,
            product_fake_percentage
        ]
    })

    st.bar_chart(
        product_percentages.set_index(
            "Review Type"
        )
    )

# ============================================================
# PRODUCT STAR RATING
# ============================================================

st.subheader(
    "⭐ Product Star Rating Distribution"
)

rating_distribution = (
    product_df["star_rating"]
    .value_counts()
    .sort_index()
)

rating_distribution.index = (
    rating_distribution.index.astype(str)
)

st.bar_chart(
    rating_distribution
)

# ============================================================
# PRODUCT SENTIMENT
# ============================================================

st.subheader(
    "😊 Product Sentiment Distribution"
)

sentiment_distribution = (
    product_df["sentiment"]
    .value_counts()
)

st.bar_chart(
    sentiment_distribution
)

# ============================================================
# PRODUCT SUSPICIOUS RATE BY STAR
# ============================================================

st.subheader(
    "🚨 Suspicious Rate by Star Rating"
)

product_star_risk = (
    product_df
    .groupby("star_rating")["is_fake_review"]
    .mean()
    .mul(100)
)

st.bar_chart(
    product_star_risk
)

# ============================================================
# PRODUCT MONTHLY TREND
# ============================================================

st.subheader(
    "📈 Product Suspicious Review Trend"
)

product_monthly = (
    product_df
    .groupby(
        ["year", "month"]
    )["is_fake_review"]
    .mean()
    .mul(100)
    .reset_index()
)

product_monthly["Period"] = (
    product_monthly["year"].astype(str)
    + "-"
    + product_monthly["month"]
    .astype(str)
    .str.zfill(2)
)

product_monthly = product_monthly.sort_values(
    ["year", "month"]
)

product_trend = product_monthly[
    ["Period", "is_fake_review"]
].set_index("Period")

product_trend.columns = [
    "Suspicious Percentage"
]

if len(product_trend) > 0:

    st.line_chart(
        product_trend
    )

else:

    st.info(
        "Not enough historical data for a trend."
    )

# ============================================================
# PRODUCT INSIGHTS
# ============================================================

st.subheader(
    "🧠 Product Insights"
)

insights = []

if product_fake_percentage >= 10:

    insights.append(
        f"⚠️ {product_fake_percentage:.1f}% of reviews "
        "are marked suspicious."
    )

else:

    insights.append(
        f"✅ Suspicious-review rate is "
        f"{product_fake_percentage:.1f}%."
    )

if product_total < 10:

    insights.append(
        "ℹ️ This product has a relatively small "
        "number of reviews."
    )

if product_fake_percentage > product_genuine_percentage:

    insights.append(
        "🚨 Suspicious reviews exceed genuine "
        "reviews for this product."
    )

if product_df["star_rating"].mean() >= 4.5:

    insights.append(
        "⭐ The product has a very high average "
        "star rating."
    )

for insight in insights:

    st.write(
        insight
    )

st.caption(
    "⚠️ Product Trust Score is based on the dataset's review labels. "
    "It is an analytical indicator and should not be treated as proof "
    "of fraudulent activity."
)

# ============================================================
# SECTION 3 — OVERALL REVIEW HEALTH
# ============================================================

st.divider()

st.header(
    "📊 Overall Review Health"
)

total_reviews = len(df)

fake_reviews = int(
    (df["is_fake_review"] == 1).sum()
)

genuine_reviews = int(
    (df["is_fake_review"] == 0).sum()
)

fake_percentage = (
    fake_reviews / total_reviews * 100
    if total_reviews > 0
    else 0
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
# OVERALL GENUINE VS SUSPICIOUS
# ============================================================

st.subheader(
    "🔎 Genuine vs Suspicious Reviews"
)

review_counts = pd.DataFrame({
    "Review Type": [
        "Genuine",
        "Suspicious"
    ],
    "Count": [
        genuine_reviews,
        fake_reviews
    ]
})

st.bar_chart(
    review_counts.set_index(
        "Review Type"
    )
)

# ============================================================
# OVERALL STAR ANALYSIS
# ============================================================

st.subheader(
    "⭐ Suspicious Reviews by Star Rating"
)

star_analysis = (
    df
    .groupby("star_rating")["is_fake_review"]
    .mean()
    .mul(100)
    .reset_index()
)

star_analysis.columns = [
    "Star Rating",
    "Suspicious Percentage"
]

st.bar_chart(
    star_analysis.set_index(
        "Star Rating"
    )
)

# ============================================================
# OVERALL SENTIMENT ANALYSIS
# ============================================================

st.subheader(
    "😊 Suspicious Reviews by Sentiment"
)

sentiment_analysis = (
    df
    .groupby("sentiment")["is_fake_review"]
    .mean()
    .mul(100)
    .reset_index()
)

sentiment_analysis.columns = [
    "Sentiment",
    "Suspicious Percentage"
]

st.bar_chart(
    sentiment_analysis.set_index(
        "Sentiment"
    )
)

# ============================================================
# OVERALL MONTHLY TREND
# ============================================================

st.subheader(
    "📈 Overall Suspicious Review Trend"
)

monthly_analysis = (
    df
    .groupby(
        ["year", "month"]
    )["is_fake_review"]
    .mean()
    .mul(100)
    .reset_index()
)

monthly_analysis["Period"] = (
    monthly_analysis["year"].astype(str)
    + "-"
    + monthly_analysis["month"]
    .astype(str)
    .str.zfill(2)
)

monthly_analysis = monthly_analysis.sort_values(
    ["year", "month"]
)

trend_data = monthly_analysis[
    ["Period", "is_fake_review"]
].set_index("Period")

trend_data.columns = [
    "Suspicious Percentage"
]

st.line_chart(
    trend_data
)

# ============================================================
# CATEGORY ANALYSIS
# ============================================================

st.subheader(
    "🛒 Suspicious Reviews by Product Category"
)

category_analysis = (
    df
    .groupby("category")["is_fake_review"]
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
    category_analysis.set_index(
        "Category"
    )
)

# ============================================================
# SECTION 4 — MODEL COMPARISON
# ============================================================

st.divider()

st.header(
    "🤖 Model Performance Comparison"
)

st.write(
    "ReviewGuard evaluates Logistic Regression and Random Forest "
    "using the same train-test split and feature preprocessing."
)

# ============================================================
# PREPARE MODEL DATA
# ============================================================

@st.cache_data
def prepare_model_data(data):

    model_df = data.copy()

    # Remove identifiers and date
    drop_columns = [
        "review_id",
        "product_id",
        "user_id",
        "review_date",
        "is_fake_review"
    ]

    X = model_df.drop(
        columns=drop_columns
    )

    y = model_df["is_fake_review"]

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# TRAIN + EVALUATE MODELS
# ============================================================

# ============================================================
# MODEL DATA PREPARATION
# ============================================================

def prepare_model_data(data):

    # Make a fresh copy to avoid read-only cached arrays
    model_df = data.copy(deep=True)

    # Remove identifiers and target from features
    drop_columns = [
        "review_id",
        "product_id",
        "user_id",
        "review_date",
        "is_fake_review"
    ]

    X = model_df.drop(
        columns=drop_columns
    ).copy()

    # IMPORTANT:
    # Create an independent writable copy of target
    y = model_df["is_fake_review"].to_numpy(
        dtype="int64",
        copy=True
    )

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(data):

    # --------------------------------------------------------
    # LOGISTIC REGRESSION DATA
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_model_data(data)

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_predictions = logistic_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    (
        X_train_rf,
        X_test_rf,
        y_train_rf,
        y_test_rf,
        preprocessor_rf
    ) = prepare_model_data(data)

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor_rf
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ]
    )

    random_forest_model.fit(
        X_train_rf,
        y_train_rf
    )

    rf_predictions = random_forest_model.predict(
        X_test_rf
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Logistic Regression": [

            accuracy_score(
                y_test,
                logistic_predictions
            ),

            precision_score(
                y_test,
                logistic_predictions,
                zero_division=0
            ),

            recall_score(
                y_test,
                logistic_predictions,
                zero_division=0
            ),

            f1_score(
                y_test,
                logistic_predictions,
                zero_division=0
            )
        ],

        "Random Forest": [

            accuracy_score(
                y_test_rf,
                rf_predictions
            ),

            precision_score(
                y_test_rf,
                rf_predictions,
                zero_division=0
            ),

            recall_score(
                y_test_rf,
                rf_predictions,
                zero_division=0
            ),

            f1_score(
                y_test_rf,
                rf_predictions,
                zero_division=0
            )
        ]
    })

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    logistic_cm = confusion_matrix(
        y_test,
        logistic_predictions
    )

    rf_cm = confusion_matrix(
        y_test_rf,
        rf_predictions
    )

    return (
        metrics,
        logistic_cm,
        rf_cm
    )
# ============================================================
# EVALUATION BUTTON
# ============================================================

run_comparison = st.button(
    "⚙️ Run Model Comparison",
    use_container_width=True
)

if run_comparison:

    with st.spinner(
        "Training and evaluating both models..."
    ):

        (
            metrics,
            logistic_cm,
            rf_cm
        ) = evaluate_models(df)

    # --------------------------------------------------------
    # METRIC TABLE
    # --------------------------------------------------------

    st.subheader(
        "📋 Model Metrics"
    )

    display_metrics = metrics.copy()

    for column in [
        "Logistic Regression",
        "Random Forest"
    ]:

        display_metrics[column] = (
            display_metrics[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # COMPARISON CHART
    # --------------------------------------------------------

    st.subheader(
        "📊 Performance Comparison"
    )

    chart_metrics = metrics.set_index(
        "Metric"
    )

    st.bar_chart(
        chart_metrics
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    st.subheader(
        "🔲 Confusion Matrices"
    )

    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:

        st.write(
            "**Logistic Regression**"
        )

        logistic_cm_df = pd.DataFrame(
            logistic_cm,
            index=[
                "Actual Genuine",
                "Actual Suspicious"
            ],
            columns=[
                "Predicted Genuine",
                "Predicted Suspicious"
            ]
        )

        st.dataframe(
            logistic_cm_df,
            use_container_width=True
        )

    with cm_col2:

        st.write(
            "**Random Forest**"
        )

        rf_cm_df = pd.DataFrame(
            rf_cm,
            index=[
                "Actual Genuine",
                "Actual Suspicious"
            ],
            columns=[
                "Predicted Genuine",
                "Predicted Suspicious"
            ]
        )

        st.dataframe(
            rf_cm_df,
            use_container_width=True
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "🧠 Model Interpretation"
    )

    st.write(
        "Both models are evaluated on the same held-out test set "
        "using identical feature preprocessing."
    )

    st.info(
        "The dataset is synthetic and contains highly separable "
        "behavioral patterns. Therefore, unusually high benchmark "
        "scores should not be interpreted as equivalent real-world "
        "performance."
    )

# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🤖 ReviewGuard Model Information"
)

st.write(
    "**Production Model:** Random Forest Classifier"
)

st.write(
    "**Detection Approach:** Behavioral fake-review detection"
)

st.write(
    "**Features:** Review characteristics, reviewer behavior, "
    "purchase information, sentiment and product metadata."
)

st.write(
    "**Output:** Suspicious/Genuine prediction + risk probability "
    "+ trust score + explainability."
)

st.caption(
    "Dataset contains synthetic e-commerce review data. "
    "Model performance should therefore be interpreted within "
    "the context of this dataset."
)

st.caption(
    "⚠️ ReviewGuard provides analytical risk indicators and "
    "does not establish that a specific review is fraudulent."
)