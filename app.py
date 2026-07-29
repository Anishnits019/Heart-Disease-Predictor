import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib  # or import pickle / load your pipeline artifact

# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Cardiovascular Disease Risk Predictor")
st.markdown(
    "Enter patient details below. The app transforms these raw inputs using your custom feature engineering pipeline before feeding them to the trained ML model."
)

st.divider()

# ==========================================
# 1. RAW USER INPUT FORM
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Objective Features")
    
    # Dataset stores age in days or years. Prompting for years is user-friendly:
    age_years = st.number_input("Age (Years)", min_value=18, max_value=100, value=50, step=1)
    
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

# ==========================================
# 2. FEATURE ENGINEERING TRANSFORM FUNCTION
# ==========================================
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the exact custom feature engineering transformations
    from your FeatureExtraction pipeline.
    """
    df = df.copy()

    # Hemodynamic / Health Metrics
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['map'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3

    # Binary Risk Indicators
    df["Hypertension"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)
    df["Age_Cholesterol"] = df["age"] * df["cholesterol"]
    df["Normal_BP"] = ((df["ap_hi"] < 120) & (df["ap_lo"] < 80)).astype(int)
    df["High_Cholesterol"] = (df["cholesterol"] > 1).astype(int)
    df["High_Glucose"] = (df["gluc"] > 1).astype(int)

    # Age Bins (Added .astype(str) here)
    bins = [29, 40, 50, 60, 70]
    labels = ['30-40', '40-50', '50-64', '60+']
    df['age_bins'] = pd.cut(df['age'], bins=bins, labels=labels).astype(str)

    # BMI Bins (Added .astype(str) here)
    bmi_bins = [0, 18.5, 25, 30, 35, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese', 'severely_obese']
    df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels).astype(str)

    # Lifestyle Score
    df['unhealthy_life_style'] = df['smoke'] + df['alco'] + (1 - df['active'])

    # Blood Pressure Category (Added .astype(str) here)
    conditions = [
        (df['ap_hi'] < 120) & (df['ap_lo'] < 80),
        (df['ap_hi'] >= 120) & (df['ap_hi'] < 130) & (df['ap_lo'] < 80),
        ((df['ap_hi'] >= 130) & (df['ap_hi'] < 140)) | ((df['ap_lo'] >= 80) & (df['ap_lo'] < 90))
    ]
    choices = ['normal', 'elevated', 'stage1']
    df['bp_category'] = np.select(conditions, choices, default='stage2').astype(str)

    return df


# ==========================================
# 3. MODEL INFERENCE & DISPLAY
# ==========================================
if st.button("🔍 Predict Cardiovascular Risk Probability", use_container_width=True, type="primary"):
    
    # Build initial raw input dataframe matching raw feature structure
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

    # Apply your Feature Extraction calculations
    processed_df = apply_feature_engineering(raw_df)

    st.markdown("### ⚙️ Processed Input Features")
    st.caption("This is the exact feature matrix constructed for the machine learning model:")
    st.dataframe(processed_df, use_container_width=True)

    # --- Load & Predict with Trained Model ---
    model_path = "model.pkl"  # Replace with your actual model file path
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # Predict probability for class 1 (cardio present)
        probability = model.predict_proba(processed_df)[0][1]
        prob_percentage = probability * 100

        st.markdown("### 📊 Prediction Result")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Disease Probability", f"{prob_percentage:.1f}%")

        with res_col2:
            if prob_percentage < 35:
                st.success("🟢 **Low Risk**: The model predicts low cardiovascular risk.")
            elif 35 <= prob_percentage < 65:
                st.warning("🟡 **Moderate Risk**: Moderate cardiovascular risk detected.")
            else:
                st.error("🔴 **High Risk**: High cardiovascular risk detected.")

        st.progress(int(prob_percentage))
    else:
        st.info("💡 **Model file not found.** Place your trained model (`model.pkl` or pipeline) in the project directory to get live predictions.")