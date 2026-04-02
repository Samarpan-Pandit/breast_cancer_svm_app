import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ================= FIX: CUSTOM CLASSES =================
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.log1p(np.abs(X))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.COLS = X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=self.COLS)
        d = X.copy()
        eps = 1e-9

        d["area_perimeter_ratio"] = d["area_mean"] / (d["perimeter_mean"] + eps)
        d["compactness_index"] = d["perimeter_mean"] ** 2 / (d["area_mean"] + eps)
        d["texture_se_x_worst"] = d["texture_se"] * d["texture_worst"]
        d["symmetry_se_x_mean"] = d["symmetry_se"] * d["symmetry_mean"]
        d["fracdim_x_concavity"] = d["fractal_dimension_mean"] * d["concavity_mean"]
        d["smoothness_se_x_area"] = d["smoothness_se"] * d["area_mean"]
        d["radius_worst_mean_ratio"] = d["radius_worst"] / (d["radius_mean"] + eps)
        d["concavity_worst_mean_ratio"] = d["concavity_worst"] / (d["concavity_mean"] + eps)

        return d.values


class NumpyToArrayToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.columns)

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# ================= LOAD FILES =================
full_pipeline = joblib.load("best_svm_pipeline.pkl")
threshold = joblib.load("best_threshold.pkl")
feature_cols = joblib.load("feature_columns.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("breast_cancer.csv")

df = load_data()

st.title("🩷 Breast Cancer Prediction System (SVM Advanced)")

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA",
    "📈 Visualizations",
    "🤖 Prediction",
    "📉 Evaluation"
])

# ================= TAB 1 =================
with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write(df.describe())

    st.subheader("Diagnosis Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="diagnosis", palette={"M": "red", "B": "green"}, ax=ax)
    st.pyplot(fig)

# ================= TAB 2 =================
with tab2:
    st.header("Feature Visualization")

    numeric_cols = df.select_dtypes(include=np.number).columns
    selected = st.selectbox("Select Feature", numeric_cols)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[selected], kde=True, ax=axes[0])
    sns.boxplot(y=df[selected], ax=axes[1])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    st.pyplot(fig)

# ================= TAB 3 =================
with tab3:
    st.header("Prediction System")

    user_input = {}
    for col in feature_cols:
        user_input[col] = st.number_input(col, value=float(df[col].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prob = full_pipeline.predict_proba(input_df)[0][1]
        pred = 1 if prob >= threshold else 0

        if pred == 1:
            st.error(f"⚠️ Malignant | Probability: {prob:.4f}")
        else:
            st.success(f"✅ Benign | Probability: {prob:.4f}")

        fig, ax = plt.subplots()
        ax.bar(["Benign", "Malignant"], [1-prob, prob])
        ax.set_title("Prediction Probability")
        st.pyplot(fig)

# ================= TAB 4 =================
with tab4:
    st.header("Model Evaluation")

    X = df[feature_cols]
    y = (df["diagnosis"] == "M").astype(int)

    y_proba = full_pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report, roc_auc_score

    fpr, tpr, _ = roc_curve(y, y_proba)
    precision, recall, _ = precision_recall_curve(y, y_proba)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set_title("ROC Curve")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_title("Precision-Recall Curve")
        st.pyplot(fig)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.write(f"ROC-AUC Score: {roc_auc_score(y, y_proba):.4f}")