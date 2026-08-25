import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config for nicer app title and icon
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stSidebar .sidebar-content {
        background-color: #e8f0fe;
        padding: 20px;
        border-radius: 10px;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load and preprocess data ---
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("Credit Score Classification Dataset.csv")
    df = df.dropna()

    # Select numeric features only for this model
    X = df.select_dtypes(include=['number'])
    y = df['Credit Score']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X, X_scaled, y, scaler

df, X, X_scaled, y, scaler = load_data()

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=7)

# Train Gradient Boosting Classifier
@st.cache_resource(show_spinner=False)
def train_model():
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=7)
    gbc.fit(X_train, y_train)
    return gbc

model = train_model()

# --- Streamlit UI ---
st.title("💳 Credit Score Prediction App")
st.markdown("""
Welcome! This app predicts the **Credit Score category** of a patient based on key input parameters like age, gender, health conditions, and smoking status.
""")

with st.expander("❓ About the Model and Dataset"):
    st.write("""
    - The model uses Gradient Boosting Classifier trained on numeric features.
    - The dataset includes patient demographic and health info to classify credit risk levels.
    - Input patient data in the sidebar and get instant predictions.
    """)

st.sidebar.header("Input Patient Data")
st.sidebar.markdown("Adjust the values below to predict the credit score category.")

def user_input_features():
    input_data = {}
    for col in X.columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        # Use checkbox for binary features, slider for continuous
        if sorted(df[col].unique()) == [0, 1]:
            input_data[col] = st.sidebar.checkbox(f"{col.replace('_', ' ').title()}", value=False)
        else:
            step = 1 if df[col].dtype == int else 0.1
            input_data[col] = st.sidebar.slider(
                f"{col.replace('_', ' ').title()}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=step
            )
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Standardize user input
input_scaled = scaler.transform(input_df)

# Predict on user input
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)

label_map = {0: "Low Credit Score", 1: "Medium Credit Score", 2: "High Credit Score"}

st.markdown("---")
st.subheader("Prediction Result")
st.markdown(
    f'<div style="background-color:#d1e7dd; padding:15px; border-radius:10px; color:#0f5132; font-size:22px;">'
    f'Predicted Credit Score Category: <strong>{label_map.get(prediction, prediction)}</strong>'
    '</div>',
    unsafe_allow_html=True,
)

st.subheader("Prediction Confidence")
proba_df = pd.DataFrame(prediction_proba, columns=[label_map[i] for i in range(len(prediction_proba[0]))])
st.bar_chart(proba_df.T)

# Optional: Show raw data
if st.checkbox("Show Raw Dataset"):
    st.write(df)

# Optional: Show correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    st.write("Correlation matrix of the dataset:")
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt.gcf())
