import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="HR Compensation Benchmarking Tool",
    page_icon="💼",
    layout="centered"
)

# Load Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load('salary_predictor_model.pkl')

try:
    model = load_model()
except Exception:
    st.error("Model file 'salary_predictor_model.pkl' not found. Please run 'train_model.py' first!")
    st.stop()

# Application UI
st.title("💼 HR Compensation Benchmarking Tool")
st.markdown("Predict dynamic market salary benchmarks using employee demographics, experience, and role parameters.")

st.divider()

st.subheader("Candidate Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

with col2:
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
    job_titles = [
        "Software Engineer", "Data Analyst", "Data Scientist", "Product Manager", 
        "Senior Manager", "Director", "Software Engineer Manager", "Senior Project Engineer",
        "Marketing Manager", "Back end Developer", "Sales Associate"
    ]
    job_title = st.selectbox("Job Title", job_titles)

# Derive Seniority Flag
is_senior = 1 if any(word in job_title.lower() for word in ['senior', 'manager', 'director', 'vp', 'lead']) else 0

st.divider()

if st.button("Predict Salary Benchmark", type="primary"):
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Experience': experience,
        'Is_Senior': is_senior
    }])

    prediction = model.predict(input_df)[0]
    margin = 3250.0  # MAE error margin

    st.success("### Benchmark Recommendation")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Estimated Salary", f"${prediction:,.2f}")
    with res_col2:
        st.metric("Benchmark Range", f"${max(0, prediction - margin):,.0f} - ${prediction + margin:,.0f}")

    st.info(f"""
    **Role Summary:**
    * **Job Function:** {job_title} ({'Senior Tier' if is_senior else 'Standard Tier'})
    * **Education:** {education} | **Experience:** {experience} years
    """)
