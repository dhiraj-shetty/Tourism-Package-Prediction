import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="dhirajshetty/tourism-model",
    filename="best_tourism_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Prediction App")
st.write("This app predicts whether a customer will purchase the Wellness Tourism Package.")
st.write("Please enter the customer details below to get a prediction.")

# Collect user input
Age = st.number_input("Age (customer's age)", min_value=18, max_value=100, value=35)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=3)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=6, value=3)
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips (per year)", min_value=1, max_value=10, value=2)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5 yrs)", min_value=0, max_value=5, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)
CityTier = st.selectbox("City Tier", [1, 2, 3])
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Build input dataframe
input_data = pd.DataFrame([{
    "Age": Age,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "CityTier": CityTier,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "ProductPitched": ProductPitched,
    "Designation": Designation,
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    if prediction == 1:
        st.success("The customer is **likely to purchase** the Wellness Tourism Package!")
    else:
        st.warning("The customer is **unlikely to purchase** the Wellness Tourism Package.")
