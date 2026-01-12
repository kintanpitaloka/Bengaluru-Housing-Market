import streamlit as st
import numpy as np
import pickle
import json

# =====================
# LOAD MODEL & DATA
# =====================
@st.cache_resource
def load_model():
    with open("bengaluru_house_price.pickle", "rb") as f:
        model = pickle.load(f)

    with open("columns.json", "r") as f:
        data = json.load(f)
        columns = data["data_columns"]

    return model, columns

lr_clf, feature_names = load_model()

# =====================
# PREDICTION FUNCTION
# =====================
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(feature_names))
    x[0], x[1], x[2] = sqft, bath, bhk

    if location in feature_names:
        x[feature_names.index(location)] = 1
    else:
        st.warning("Lokasi belum dikenali, prediksi pakai baseline")

    return lr_clf.predict([x])[0]

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="🏡 Bengaluru Housing Price", layout="centered")

st.title("🏡 House Price Prediction App")
st.caption("Powered by ML · Built with Streamlit")

locations = feature_names[3:]

location = st.selectbox("📍 Location", locations)
sqft = st.number_input("📐 Total Square Feet", min_value=300, step=50)
bath = st.number_input("🛁 Bathrooms", min_value=1, step=1)
bhk = st.number_input("🛏️ BHK", min_value=1, step=1)

if st.button("🚀 Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {round(price, 2)} Lakhs")
