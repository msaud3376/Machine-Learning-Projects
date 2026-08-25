import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample training function (to be run once before deployment)
@st.cache_resource
def train_model():
    df = pd.read_csv("Food_Delivery_Times.csv")
    df.dropna(inplace=True)

    categorical = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    numerical = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
    target = 'Delivery_Time_min'

    X = df[categorical + numerical]
    y = df[target]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), categorical),
        ('num', StandardScaler(), numerical)
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', LinearRegression())
    ])

    pipeline.fit(X, y)
    return pipeline, categorical, numerical

model, categorical_cols, numerical_cols = train_model()

# Streamlit App Layout
st.title("🚴‍♂️ Food Delivery Time Predictor")
st.markdown("### Enter delivery details below:")

# Input form
weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy", "Stormy", "Windy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car"])
distance = st.slider("Distance (km)", 0.5, 30.0, 5.0)
prep_time = st.slider("Preparation Time (min)", 1, 60, 15)
experience = st.slider("Courier Experience (years)", 0.0, 10.0, 2.0)

# Predict button
if st.button("Predict Delivery Time"):
    input_df = pd.DataFrame([{
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
        "Distance_km": distance,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"🍔 Your food will arrive in approximately **{prediction:.2f} minutes**! Enjoy! 🎉")