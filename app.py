import streamlit as st
import joblib
import numpy as np

# Load the model
def load_model():
    with open('saved_steps.joblib', 'rb') as file:
        data = joblib.load(file)
    return data

data = load_model()
regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Prediction UI
st.title("Software Developer Salary Prediction")
st.write("""### Enter details to predict salary""")

# Input options
countries = (
    "United States", "India", "United Kingdom", "Germany", "Canada",
    "Brazil", "France", "Spain", "Australia", "Netherlands",
    "Poland", "Italy", "Russian Federation", "Sweden",
)

education_levels = (
    "Less than a Bachelors", "Bachelor’s degree",
    "Master’s degree", "Post grad",
)

country = st.selectbox("Country", countries)
education = st.selectbox("Education Level", education_levels)
experience = st.slider("Years of Experience", 0, 50, 3)
age = st.slider("Age", 18, 70, 25)

# Predict button
if st.button("Calculate Salary"):
    X = np.array([[country, education, experience, age]])
    X[:, 0] = le_country.transform(X[:, 0])
    X[:, 1] = le_education.transform(X[:, 1])
    X = X.astype(float)

    salary = regressor.predict(X)
    st.subheader(f"Estimated Salary: **${salary[0]:,.2f}**")