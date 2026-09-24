import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ReviewGuard",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini client
gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        gemini_client = None


# ============================================================
# LOAD DATA + MODELS
# ============================================================

DATA_PATH = "Data/ecommerce_reviews.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    logistic = joblib.load("models/logistic_model.pkl")
    random_forest = joblib.load("models/random_forest_model.pkl")

    return logistic, random_forest


df = load_data()
logistic_model, random_forest_model = load_models()


# ============================================================
# SESSION STATE
# ============================================================

if "current_review" not in st.session_state:
    st.session_state.current_review = None

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
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
# HELPER FUNCTIONS
# ============================================================

def make_feature_dataframe(review_data):
    return pd.DataFrame([review_data])


def get_prediction(review_data):
    input_df = make_feature_dataframe(review_data)

    lr_prediction = logistic_model.predict(input_df)[0]
    rf_prediction = random_forest_model.predict(input_df)[0]

    try:
        lr_probability = logistic_model.predict_proba(input_df)[0][1]
    except Exception:
        lr_probability = float(lr_prediction)

    try:
        rf_probability = random_forest_model.predict_proba(input_df)[0][1]
    except Exception:
        rf_probability = float(rf_prediction)

    # Average the two model probabilities
    fake_risk = ((lr_probability + rf_probability) / 2) * 100

    if fake_risk >= 70:
        classification = "Suspicious"
    elif fake_risk <= 30:
        classification = "Likely Genuine"
    else:
        classification = "Uncertain"

    return {
        "classification": classification,
        "fake_risk": fake_risk,
        "genuine_chance": 100 - fake_risk,
        "lr_prediction": int(lr_prediction),
        "rf_prediction": int(rf_prediction),
        "lr_probability": lr_probability * 100,
        "rf_probability": rf_probability * 100
    }


def generate_explanation(review):
    signals = []

    if review["exclamation_marks"] >= 3:
        signals.append(
            f"High number of exclamation marks ({review['exclamation_marks']})"
        )

    if review["all_caps_ratio"] >= 20:
        signals.append(
            f"High capitalization ratio ({review['all_caps_ratio']:.1f}%)"
        )

    if review["is_early_review"] == 1:
        signals.append("The review was posted very early")

    if review["reviewer_review_count"] <= 2:
        signals.append(
            f"Low reviewer history ({review['reviewer_review_count']} previous reviews)"
        )

    if review["verified_purchase"] == 0:
        signals.append("Purchase is not verified")

    if review["star_rating"] == 5:
        signals.append("Maximum 5-star rating")

    if review["sentiment"] == "positive":
        signals.append("Strong positive sentiment")

    if review["review_length_words"] <= 10:
        signals.append("Very short review")

    if not signals:
        signals.append("No major suspicious behavioral signal detected")

    return signals


# ============================================================
# LOCAL AGENT TOOLS
# ============================================================

def agent_overall_analysis():
    total = len(df)
    suspicious = int(df["is_fake_review"].sum())
    genuine = total - suspicious

    suspicious_rate = (suspicious / total) * 100

    return {
        "tool": "overall_analysis",
        "total_reviews": total,
        "suspicious_reviews": suspicious,
        "genuine_reviews": genuine,
        "suspicious_rate": suspicious_rate
    }


def agent_product_analysis(product_id):
    product_df = df[
        df["product_id"].astype(str).str.upper()
        == str(product_id).upper()
    ]

    if product_df.empty:
        return {
            "tool": "product_analysis",
            "error": f"Product {product_id} was not found."
        }

    total = len(product_df)
    suspicious = int(product_df["is_fake_review"].sum())
    genuine = total - suspicious

    return {
        "tool": "product_analysis",
        "product_id": product_id,
        "total_reviews": total,
        "suspicious_reviews": suspicious,
        "genuine_reviews": genuine,
        "suspicious_rate": (suspicious / total) * 100,
        "average_rating": product_df["star_rating"].mean()
    }


def agent_category_analysis(category):
    category_df = df[
        df["category"].astype(str).str.lower()
        == str(category).lower()
    ]

    if category_df.empty:
        return {
            "tool": "category_analysis",
            "error": f"Category '{category}' was not found."
        }

    total = len(category_df)
    suspicious = int(category_df["is_fake_review"].sum())

    return {
        "tool": "category_analysis",
        "category": category,
        "total_reviews": total,
        "suspicious_reviews": suspicious,
        "genuine_reviews": total - suspicious,
        "suspicious_rate": (suspicious / total) * 100,
        "average_rating": category_df["star_rating"].mean()
    }


def agent_category_comparison(category1, category2):
    result1 = agent_category_analysis(category1)
    result2 = agent_category_analysis(category2)

    return {
        "tool": "category_comparison",
        "category1": result1,
        "category2": result2
    }


def agent_highest_risk_category():
    category_stats = (
        df.groupby("category")["is_fake_review"]
        .agg(["count", "sum"])
        .reset_index()
    )

    category_stats["suspicious_rate"] = (
        category_stats["sum"] / category_stats["count"] * 100
    )

    row = category_stats.loc[
        category_stats["suspicious_rate"].idxmax()
    ]

    return {
        "tool": "highest_risk_category",
        "category": row["category"],
        "suspicious_rate": row["suspicious_rate"],
        "total_reviews": row["count"],
        "suspicious_reviews": row["sum"]
    }


def agent_explain_current_review():
    review = st.session_state.current_review

    if review is None:
        return {
            "tool": "review_explanation",
            "error": "No review has been analyzed yet."
        }

    prediction = get_prediction(review)
    signals = generate_explanation(review)

    return {
        "tool": "review_explanation",
        "classification": prediction["classification"],
        "fake_risk": prediction["fake_risk"],
        "genuine_chance": prediction["genuine_chance"],
        "signals": signals
    }


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(query):

    q = query.lower().strip()

    # Current review explanation
    if (
        "why" in q
        and (
            "suspicious" in q
            or "fake" in q
            or "risk" in q
        )
    ):
        return "explain_review"

    if "show suspicious reviews" in q or "suspicious reviews for" in q:
        return "product_analysis"

    if (
        "how many suspicious" in q
        and ("product" in q or "reviews" in q)
    ):
        return "product_analysis"

    if (
        "average rating" in q
        or "rating of" in q
    ):
        return "product_analysis"

    if (
        "suspicious rate" in q
        and any(
            category.lower() in q
            for category in df["category"].unique()
        )
    ):
        return "category_analysis"

    if "compare" in q:
        return "comparison"

    if (
        "highest risk" in q
        or "most suspicious category" in q
        or "riskiest category" in q
    ):
        return "highest_risk"

    if (
        "overall" in q
        or "total reviews" in q
        or "how many reviews" in q
    ):
        return "overall"

    return "general"


# ============================================================
# EXTRACT PRODUCT ID
# ============================================================

def extract_product_id(query):

    match = re.search(
        r"(P\d+)",
        query.upper()
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# EXTRACT CATEGORY
# ============================================================

def extract_categories(query):

    found = []

    q = query.lower()

    for category in df["category"].dropna().unique():

        if str(category).lower() in q:
            found.append(category)

    return found


# ============================================================
# RUN LOCAL TOOL
# ============================================================

def run_agent_tool(query):

    intent = detect_intent(query)

    if intent == "explain_review":
        return agent_explain_current_review()

    if intent == "product_analysis":

        product_id = extract_product_id(query)

        if product_id:
            return agent_product_analysis(product_id)

        return {
            "tool": "product_analysis",
            "error": "Please provide a product ID such as P000001."
        }

    if intent == "category_analysis":

        categories = extract_categories(query)

        if categories:
            return agent_category_analysis(categories[0])

        return {
            "tool": "category_analysis",
            "error": "Please specify a valid category."
        }

    if intent == "comparison":

        categories = extract_categories(query)

        if len(categories) >= 2:
            return agent_category_comparison(
                categories[0],
                categories[1]
            )

        return {
            "tool": "category_comparison",
            "error": "Please specify two valid categories."
        }

    if intent == "highest_risk":
        return agent_highest_risk_category()

    if intent == "overall":
        return agent_overall_analysis()

    return {
        "tool": "general",
        "message": "I can analyze reviews, products, categories and suspicious-review patterns."
    }


# ============================================================
# GEMINI RESPONSE GENERATOR
# ============================================================

def generate_gemini_response(user_query, tool_result):

    if gemini_client is None:
        return None

    system_context = """
You are ReviewGuard AI Agent.

ReviewGuard is an AI-powered e-commerce review trust and risk analyzer.

IMPORTANT RULES:
1. Use ONLY the factual information provided in TOOL RESULT.
2. Never invent statistics, product IDs, percentages, ratings or review counts.
3. Do not claim that a review is definitely fake.
4. Use phrases such as "suspicious", "likely suspicious", or "risk indicator".
5. Keep answers concise and useful.
6. Explain technical results in simple language.
7. ReviewGuard uses behavioral review signals and ML models.
"""

    prompt = f"""
{system_context}

USER QUERY:
{user_query}

TOOL RESULT:
{tool_result}

Generate a clear answer to the user.
"""

    try:
        interaction = gemini_client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return interaction.output_text

    except Exception as e:
        return f"Gemini response failed. Local analysis result: {tool_result}"


# ============================================================
# AGENT EXECUTION
# ============================================================

def ask_agent(query):

    tool_result = run_agent_tool(query)

    gemini_response = generate_gemini_response(
        query,
        tool_result
    )

    if gemini_response:
        answer = gemini_response
    else:
        answer = str(tool_result)

    st.session_state.agent_history.append(
        {
            "user": query,
            "answer": answer
        }
    )

    return answer


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ ReviewGuard")

st.markdown(
    """
### AI-Powered Review Trust & Risk Analyzer

Detect suspicious review patterns, understand the risk,
analyze products and interact with the ReviewGuard AI Agent.
"""
)

st.divider()


# ============================================================
# 🤖 AI AGENT — TOP
# ============================================================

st.subheader("🤖 ReviewGuard AI Agent")

if gemini_client:
    st.success("Gemini AI connected")
else:
    st.warning(
        "Gemini API is not connected. Local Agent tools are still available."
    )


agent_query = st.text_input(
    "Ask ReviewGuard anything",
    placeholder="Example: Why is this review suspicious?"
)

if st.button("🤖 Ask Agent", use_container_width=True):

    if agent_query.strip():

        with st.spinner("ReviewGuard Agent is analyzing..."):

            answer = ask_agent(agent_query)

        st.markdown("### Agent Response")
        st.write(answer)

    else:
        st.warning("Please enter a question.")


# Chat history

if st.session_state.agent_history:

    with st.expander("💬 Agent Conversation History"):

        for item in st.session_state.agent_history:

            st.markdown(
                f"**You:** {item['user']}"
            )

            st.markdown(
                f"**🤖 ReviewGuard:** {item['answer']}"
            )

            st.divider()


st.caption(
    "Agent workflow: User Query → Intent Detection → Tool Selection → "
    "Data Analysis → Gemini Response"
)

st.divider()


# ============================================================
# SECTION 1 — SINGLE REVIEW ANALYZER
# ============================================================

st.header("🔍 1. Single Review Analyzer")

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
        sorted(df["sentiment"].dropna().unique())
    )

    verified_purchase = st.selectbox(
        "Verified Purchase",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:

    review_length_words = st.number_input(
        "Review Length (words)",
        min_value=1,
        value=25
    )

    title_word_count = st.number_input(
        "Title Word Count",
        min_value=0,
        value=4
    )

    num_images_attached = st.number_input(
        "Images Attached",
        min_value=0,
        value=0
    )

    helpful_votes = st.number_input(
        "Helpful Votes",
        min_value=0,
        value=0
    )

    total_votes = st.number_input(
        "Total Votes",
        min_value=1,
        value=1
    )

    has_title = st.selectbox(
        "Has Title",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:

    days_since_purchase = st.number_input(
        "Days Since Purchase",
        min_value=0,
        value=2
    )

    reviewer_review_count = st.number_input(
        "Reviewer Review Count",
        min_value=0,
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
        value=6
    )

    all_caps_ratio = st.number_input(
        "All Caps Ratio (%)",
        min_value=0.0,
        value=25.0
    )

    readability_score = st.number_input(
        "Readability Score",
        min_value=0.0,
        value=70.0
    )


col4, col5, col6 = st.columns(3)

with col4:
    category = st.selectbox(
        "Category",
        sorted(df["category"].dropna().unique())
    )

with col5:
    price_usd = st.number_input(
        "Price (USD)",
        min_value=0.0,
        value=49.99
    )

with col6:

    price_tier = st.selectbox(
        "Price Tier",
        sorted(df["price_tier"].dropna().unique())
    )

brand_tier = st.selectbox(
    "Brand Tier",
    sorted(df["brand_tier"].dropna().unique())
)


if st.button(
    "🔎 Analyze Review",
    type="primary",
    use_container_width=True
):

    helpful_ratio = (
        helpful_votes / total_votes
        if total_votes > 0
        else 0
    )

    current_review = {
        "year": year,
        "month": month,
        "star_rating": star_rating,
        "sentiment": sentiment,
        "verified_purchase": verified_purchase,
        "review_length_words": review_length_words,
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
    }

    st.session_state.current_review = current_review

    prediction = get_prediction(current_review)
    signals = generate_explanation(current_review)

    st.subheader("Prediction")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Classification",
            prediction["classification"]
        )

    with result_col2:
        st.metric(
            "Fake Risk",
            f"{prediction['fake_risk']:.2f}%"
        )

    with result_col3:
        st.metric(
            "Genuine Chance",
            f"{prediction['genuine_chance']:.2f}%"
        )

    st.subheader("🧠 Why this prediction?")

    for signal in signals:
        st.write(f"⚠️ {signal}")

    st.info(
        "Risk score is an indicator of suspicious patterns, "
        "not proof that a review is fake."
    )


# ============================================================
# SECTION 2 — PRODUCT REVIEW HEALTH
# ============================================================

st.header("📦 2. Product Review Health")

selected_product = st.selectbox(
    "Select Product",
    sorted(df["product_id"].dropna().unique())
)

product_df = df[
    df["product_id"] == selected_product
]

if not product_df.empty:

    total_reviews = len(product_df)
    suspicious_reviews = int(
        product_df["is_fake_review"].sum()
    )
    genuine_reviews = total_reviews - suspicious_reviews

    suspicious_rate = (
        suspicious_reviews / total_reviews * 100
    )

    avg_rating = product_df["star_rating"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Reviews",
        total_reviews
    )

    c2.metric(
        "Genuine Reviews",
        genuine_reviews
    )

    c3.metric(
        "Suspicious Reviews",
        suspicious_reviews
    )

    c4.metric(
        "Average Rating",
        f"{avg_rating:.2f} ⭐"
    )

    st.progress(
        min(suspicious_rate / 100, 1.0)
    )

    st.caption(
        f"Suspicious Review Rate: {suspicious_rate:.2f}%"
    )


# ============================================================
# SECTION 3 — OVERALL REVIEW HEALTH
# ============================================================

st.header("📊 3. Overall Review Health")

total_reviews = len(df)
total_suspicious = int(df["is_fake_review"].sum())
total_genuine = total_reviews - total_suspicious

overall_rate = (
    total_suspicious / total_reviews * 100
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Reviews",
    f"{total_reviews:,}"
)

c2.metric(
    "Genuine Reviews",
    f"{total_genuine:,}"
)

c3.metric(
    "Suspicious Reviews",
    f"{total_suspicious:,}"
)

st.metric(
    "Overall Suspicious Rate",
    f"{overall_rate:.2f}%"
)


# ============================================================
# SECTION 4 — MODEL PERFORMANCE COMPARISON
# ============================================================

st.header("⚖️ 4. Model Performance Comparison")

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        1.00,
        1.00
    ],
    "Precision": [
        1.00,
        1.00
    ],
    "Recall": [
        1.00,
        1.00
    ],
    "F1 Score": [
        1.00,
        1.00
    ]
})

st.dataframe(
    model_results,
    use_container_width=True
)

st.info(
    "Both models achieved 100% on the held-out test split of the "
    "synthetic dataset. This should not be interpreted as real-world "
    "accuracy."
)


# ============================================================
# SECTION 5 — MODEL INFORMATION
# ============================================================

st.header("ℹ️ 5. Model Information")

st.markdown(
    """
### Models Used

**Logistic Regression**
- Linear classification model
- Provides a strong interpretable baseline

**Random Forest**
- Ensemble of decision trees
- Captures nonlinear feature relationships

### Important Features

- Exclamation marks
- Capitalization ratio
- Star rating
- Sentiment
- Reviewer history
- Early-review behavior
- Purchase verification
- Review engagement

### Architecture

**Data → Preprocessing → ML Models → Prediction → Risk Score → Explanation → Agent**
"""
)

st.caption(
    "ReviewGuard — AI Review Trust & Risk Analyzer"
)