# 🛡️ ReviewGuard — AI Review Trust & Risk Analyzer

ReviewGuard is an AI-powered system designed to identify potentially suspicious e-commerce reviews using behavioral and review-level signals.

Instead of simply labeling a review as fake or genuine, ReviewGuard provides a risk score and interpretable signals to help users understand why a review may require further attention.

## 🚀 Features

- 🔍 Suspicious review detection
- 🤖 Machine Learning based classification
- 🌲 Random Forest model
- 📈 Logistic Regression model comparison
- 🎯 Fake-review risk score
- 🧠 Explainable behavioral warning signals
- 📊 Review-level analysis
- 🖥️ Interactive Streamlit interface
- 💾 Trained models saved using Joblib

## 💡 Problem Statement

Fake and manipulated reviews can affect customer decisions and make it difficult to identify trustworthy products.

ReviewGuard addresses this problem by analyzing behavioral characteristics associated with suspicious reviews and generating a risk-based prediction.

## 🧠 How It Works

User Review Information
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning Models
     ↙              ↘
Logistic Regression  Random Forest
     ↘              ↙
    Risk Prediction
        ↓
Explanation & Insights
        ↓
   Streamlit Dashboard

## 📊 Dataset

The primary dataset contains 100,000 e-commerce reviews with multiple behavioral and review-level attributes.

Important features include:

- Star rating
- Sentiment
- Verified purchase
- Review length
- Number of images
- Reviewer history
- Days since purchase
- Exclamation marks
- Capitalization ratio
- Readability score
- Product category
- Product price
- Brand tier
- Price tier

### Target Variable

`is_fake_review`

- `0` → Genuine
- `1` → Fake / Suspicious

> Note: The primary dataset is synthetic and was generated to model realistic e-commerce review distributions. Therefore, model performance on this dataset should not be interpreted as real-world accuracy.

## 🤖 Machine Learning Models

ReviewGuard currently compares two classification approaches.

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest

An ensemble-based classifier used to capture non-linear relationships between review behavioral features.

Both models use preprocessing pipelines containing:

- Missing-value imputation
- Standardization for numerical features
- One-hot encoding for categorical features

## 📈 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

### Current Benchmark

On the synthetic benchmark dataset, both models achieved:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 100% | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% | 100% |

> ⚠️ These results reflect the characteristics of the synthetic dataset. The dataset contains strong behavioral signals associated with its generated labels, so these results should not be treated as evidence of 100% real-world fake-review detection accuracy.

## 🧠 Explainability

ReviewGuard highlights behavioral signals that may contribute to a suspicious-review prediction, such as:

- 🔠 High capitalization usage
- ❗ Excessive exclamation marks
- ⭐ Extreme star ratings
- ⚡ Very early reviews
- 🛒 Unverified purchases
- 📝 Very short reviews
- 👤 Limited reviewer history

These signals are presented as indicators, not proof that a review is fake.

## 🖥️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Joblib | Model persistence |
| Streamlit | Interactive UI |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |

## 📁 Project Structure

```text
ReviewGuard/
│
├── Data/
│   ├── ecommerce_reviews.csv
│   ├── monthly_trends.csv
│   ├── products.csv
│   ├── sellers.csv
│   └── sentiment_labels.csv
│
├── models/
│   ├── logistic_model.pkl
│   └── random_forest_model.pkl
│
├── app.py
├── predict.py
├── train_model.py
├── inspect_data.py
├── analyze_features.py
├── requirements.txt
├── .gitignore
└── README.md
