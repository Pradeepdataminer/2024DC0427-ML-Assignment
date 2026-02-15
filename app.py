import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Credit Card Default Prediction")
st.write("Machine Learning Assignment 2 – BITS Pilani")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload Credit Card Default Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # Preprocessing
    # ------------------------------
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------------------------
    # Model Selection
    # ------------------------------
    model_name = st.selectbox(
        "Select Machine Learning Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()

    elif model_name == "kNN":
        model = KNeighborsClassifier()

    elif model_name == "Naive Bayes":
        model = GaussianNB()

    elif model_name == "Random Forest":
        model = RandomForestClassifier()

    elif model_name == "XGBoost":
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss"
        )

    # ------------------------------
    # Train model
    # ------------------------------
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ------------------------------
    # Metrics
    # ------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = 0.0

    # ------------------------------
    # Display Metrics
    # ------------------------------
    st.subheader("Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("AUC", round(auc, 4))
    col3.metric("MCC", round(mcc, 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("Precision", round(precision, 4))
    col5.metric("Recall", round(recall, 4))
    col6.metric("F1 Score", round(f1, 4))

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ------------------------------
    # Classification Report
    # ------------------------------
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload a CSV file to continue.")
