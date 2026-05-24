"""
Streamlit App: Wellness Tourism Package Predictor
-------------------------------------------------
Loads the trained model and label encoders directly form the
Hugging Face Model Hub, collects customer inputs, and predicts whether
the customer is likely to purchase the Wellness Tourism Package.
"""

import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---- Configure ----
HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-model"
HF_TOKEN = os.environ.get("HF_TOKEN")

# ---- Load model & encoders from Hugging Face Model Hub ----
@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID, filename="best_model.pkl", token=HF_TOKEN
    )
    encoders_path = hf_hub_download(
        repo_id=MODEL_REPO_ID, filename="label_encoders.pkl", token=HF_TOKEN
    )
    return joblib.load(model_path), joblib.load(encoders_path)

model, label_encoders = load_artifacts()

# Training-time feature order (sklearn 1.5+ checks feature_names_in_ at predit time).
# Fall back to a hardcoded list only if hte attribute is missing.
FEATURE_ORDER = list(getattr(model, "feature_names_in_", [
    "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation",
    "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched", 
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport", 
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", 
    "Designation", "MonthlyIncome"
]))

# ---- UI ----
st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Wellness Tourism Package Predictor")
st.markdown("Predict whether a customer will purchase the **Wellness Tourism Package**.")
st.markdown("---")

# Sidebar: customer details
st.sidebar.header("Customer Details")
age = st.sidebar.slider("Age", 18, 70, 35)
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Buisiness"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
num_persons_visiting = st.sidebar.slider("Number of Persons Visisting", 1, 5, 2)
preferred_property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
num_trips = st.sidebar.slider("Number of Trips (Annual)", 1, 10, 3)
passport = st.sidebar.selectbox("Has Passport?", [0, 1])
own_car = st.sidebar.selectbox("Owns Car?", [0, 1])
num_children_visiting = st.sidebar.slider("Number of Children Visiting", 0, 3, 0)
designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.sidebar.number_input("Monthly Income", min_value=5000, max_value=100000, value=25000)

st.sidebar.header("Interaction Details")
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5 ,3)
product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
num_followups = st.sidebar.slider("Number of Follow-Ups", 1, 6, 3)
duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", 5, 60, 15)

# Encode categorical input matching trainign time encoding
def encode_input(value, column_name):
    le = label_encoders[column_name]
    if value in le.classes_:
        return le.transform([value])[0]
    return 0

# Get inputs and save them into a dataframe
raw = {
    "Age": age,
    "TypeofContact": encode_input(type_of_contact, "TypeofContact"),
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": encode_input(occupation, "Occupation"),
    "Gender": encode_input(gender, "Gender"),
    "NumberOfPersonVisiting": num_persons_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": encode_input(product_pitched, "ProductPitched"),
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": encode_input(marital_status, "MaritalStatus"),
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children_visiting,
    "Designation": encode_input(designation, "Designation"),
    "MonthlyIncome": monthly_income
}
input_data = pd.DataFrame([{col: raw[col] for col in FEATURE_ORDER}])

st.markdown("### Prediction Result")
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("✅ The customer is **likely to purchase** the Wellness Tourism Package.")
    else:
        st.warning("❌ The customer is **unlikely to purchase** the package.")

    st.metric("Purchase Probabilty", f"{probability[1]*100:.1f}%")

    st.markdown("---")
    st.markdown("#### Input Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}))
