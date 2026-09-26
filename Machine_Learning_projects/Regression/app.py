import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(
    open(
        r"C:\Users\raymo\NareshIT_DataScience_AI\Machine_Learning_Projects\Regression\linear_regression_model.pkl",
        "rb"
    )
)

st.title("Salary Prediction App")

st.write(
    "This app predicts the salary based on years of experience "
    "using Simple Linear Regression."
)

years_experience = st.number_input(
    "Enter years of Experience:",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5
)

if st.button("Predict Salary"):

    experience_input = np.array([[years_experience]])

    prediction = model.predict(experience_input)

    st.success(
        f"The predicted salary of {years_experience} years of experience "
        f"is ₹{prediction[0]:,.2f}"
    )

st.write(
    "The model was trained using a dataset of salaries and years of experience."
)