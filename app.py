import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json


# =====================
# LOAD MODEL & DATA
# =====================
with open("bengaluru_house_price.pkl", "rb") as f:
    lr_clf = pickle.load(f)

with open("columns.json", "r") as f:
    feature_names = json.load(f)
    
# =====================
# PREDICTION FUNCTION
# =====================
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(feature_names))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in feature_names:
        loc_index = feature_names.index(location)
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

st.title("🏡 House Price Prediction App")
st.caption("Powered by ML · Built with Streamlit")

locations = feature_names[3:]

location = st.selectbox("📍 Location", locations)
sqft = st.number_input("📐 Total Square Feet", min_value=300)
bath = st.number_input("🛁 Bathrooms", min_value=1)
bhk = st.number_input("🛏️ BHK", min_value=1)

if st.button("🚀 Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {round(price, 2)} Lakhs")
