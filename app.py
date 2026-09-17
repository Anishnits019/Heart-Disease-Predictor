from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
)

st.title("Cardiovascular Disease Risk Predictor")
st.markdown("Estimate cardiovascular disease risk using the trained CatBoost model.")

ROOT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():
    # Download the files from your public Hugging Face repo
    model_path = hf_hub_download(repo_id="Anishnits-4567/HD_MODEL", filename="model.pkl")
    scaler_path = hf_hub_download(repo_id="Anishnits-4567/HD_MODEL", filename="preprocessing.pkl")
    
    # Load them using pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

# Call the function to load your models
model, scaler = load_models()

st.divider()

# ==========================================
# 1. RAW USER INPUT FORM
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Objective Features")
    
    age_years = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=50.0, step=0.1)
    
    gender_str = st.radio("Gender", options=["Female", "Male"])
    gender = 1 if gender_str == "Female" else 2

    height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165, step=1)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

with col2:
    st.subheader("🩺 Examination Features")
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=240, value=120, step=1)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=160, value=80, step=1)

    cholesterol_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above Normal (3)": 3}
    cholesterol = cholesterol_map[st.selectbox("Cholesterol Level", options=list(cholesterol_map.keys()))]

    gluc_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above Normal (3)": 3}
    gluc = gluc_map[st.selectbox("Glucose Level", options=list(gluc_map.keys()))]

with col3:
    st.subheader("🌿 Lifestyle Features")
    smoke = 1 if st.radio("Do you smoke?", options=["No", "Yes"]) == "Yes" else 0
    alco = 1 if st.radio("Alcohol Intake?", options=["No", "Yes"]) == "Yes" else 0
    active = 1 if st.radio("Physically Active?", options=["Yes", "No"]) == "Yes" else 0

st.divider()

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["bmi"] = engineered["weight"] / (engineered["height"] / 100) ** 2
    engineered["bmi_category"] = pd.cut(
        engineered["bmi"],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=["underweight", "normal", "overweight", "obese", "severely_obese"],
    ).astype(str)
    return engineered[
        [
            "age", "height", "weight", "ap_hi", "ap_lo", "bmi",
            "gender", "cholesterol", "gluc", "smoke", "alco", "active",
            "bmi_category",
        ]
    ]


# ==========================================
# 3. MODEL INFERENCE & DISPLAY
# ==========================================
if st.button("🔍 Predict Cardiovascular Risk Probability", use_container_width=True, type="primary"):
    
    # Build initial raw input dataframe
    raw_df = pd.DataFrame([{
        'age': age_years,  # Pass age in years (or age_years * 365 if model trained on days)
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }])

    processed_df = apply_feature_engineering(raw_df)
    try:
        model, preprocessor = load_artifacts()
        transformed_input = preprocessor.transform(processed_df)
        probability = float(model.predict_proba(transformed_input)[0, 1])
    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
        st.stop()

    prob_percentage = probability * 100
    st.markdown("### Prediction Result")
    result_col, detail_col = st.columns([1, 2])
    with result_col:
        st.metric("Disease probability", f"{prob_percentage:.1f}%")
    with detail_col:
        if prob_percentage < 35:
            st.success("Low risk according to the model.")
        elif prob_percentage < 65:
            st.warning("Moderate risk according to the model.")
        else:
            st.error("High risk according to the model.")
    st.progress(probability)
    with st.expander("View model input"):
        st.dataframe(processed_df, use_container_width=True, hide_index=True)