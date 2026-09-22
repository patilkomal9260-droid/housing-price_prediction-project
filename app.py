import streamlit as st
import numpy as np
import joblib
from pymongo import MongoClient

# Mongodb connection setup 
client =MongoClient("mongodb://localhost:27017/")

db = client["housing_db"]
collection = db["predictions"]

# Load the trained machine learning model
model = joblib.load('house_price_model.pkl')

# Page Configuration
st.set_page_config(page_title="House Price Predictor", layout="centered")

# App Header
st.title("🏡 House Price Prediction System")
st.write("Enter the house details below to get an estimated market price:")

# Input Form Features
area = st.number_input("Area (in Sq.Ft):", min_value=300, max_value=10000, value=1000, step=50)
bedrooms = st.slider("Number of Bedrooms:", min_value=1, max_value=10, value=2)
age = st.number_input("Building Age (in Years):", min_value=0, max_value=100, value=5)

# Prediction Logic
if st.button("Predict Price"):
    input_features = np.array([[area,bedrooms,age]])
    prediction = model.predict(input_features)[0]
    st.success(f"Predicted Price: {prediction}")
    
    data = {
        "area": float(area), 
        "bedrooms": int(bedrooms),
        "age": int(age)
        }
    
    collection.insert_one(data)
    st.write("Data inserted!")  
    
    # Display Result
    st.success(f"💰 Estimated House Price: **₹ {prediction:.2f} Lakhs**")
    
    # Model Output Interpretation
    st.info("💡 **Interpretation:** House prices generally increase with larger square footage and decrease as the building age increases.")