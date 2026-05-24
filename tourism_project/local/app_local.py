"""
Local Streamlit app that loads the model and encoders from the local filesystem
instead of the Hugging Face Hub. This allows you to run the app without internet access, and is useful for testing and development.

Run from the root of the project with:
    streamlit run tourism_project/local/app_local.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
ARTIFACTS_DIR = HERE / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoders.pkl"

@st.cache_resource
def load_model_and_encoders():
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        st.error(f"Model or label encoder not found in {ARTIFACTS_DIR}. "
                 f"Run the pipeline script to generate them first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder

model, label_encoders = load_model_and_encoders()

FEATURE_ORDER = list(getattr(model, "feature_names_in_", [
    "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation",
    "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched", 
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport", 
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", 
    "Designation", "MonthlyIncome"
]))  # Get feature order from model if available

st.set_page_config(page_title="Tourism Package Predictor (Local)", page_icon="✈️", layout="centered")
st.title("✈️ Wellness Tourism Package Predictor - Local Version")
st.markdown(f"Loaded module from `{MODEL_PATH.relative_to(HERE)}` with label encoder from `{LABEL_ENCODER_PATH.relative_to(HERE)}`.")
st.markdown("---")

# Sidebar: customer details
st.sidebar.header("Customer Details")
age = st.sidebar.slider("Age", 18, 70, 35)
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
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
        return int(le.transform([value])[0])
    st.warning(f"Warning: '{value}' not seen in training for column '{column_name}'. Encoding as 0."
               f"Known values: {list(le.classes_)}. Failing back to 0 may lead to inaccurate predictions.")
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

input_data = pd.DataFrame([raw], columns=FEATURE_ORDER)

st.markdown("### Prediction Result")
if st.button("🔍 Predict"):
    st.markdown("### Input sent to model")
    st.write("The following input data was sent to the model for prediction:")
    st.dataframe(input_data)
    expected_columns = list(getattr(model, "feature_names_in_", []))
    if expected_columns and list(input_data.columns) != expected_columns:
        st.error(f"Error: Model expects features in the following order: {expected_columns}. "
                 f"Current input order: {list(input_data.columns)}. Please check the input data formatting.")

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