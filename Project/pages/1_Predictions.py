import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
clf_model = joblib.load(r'files_app\classification_model.pkl')
reg_model = joblib.load(r'files_app\regression_model.pkl')
scaler = joblib.load(r'files_app\scaler.pkl')
label_encoders = joblib.load(r'files_app\label_encoders.pkl')

st.title("🔍 Prediction Center")

tab1, tab2 = st.tabs(["🧠 Classification: Treatment Prediction", "📈 Regression: Age Prediction"])

# ======== TAB 1: CLASSIFICATION ========
with tab1:
    st.subheader("🧠 Will the individual seek mental health treatment?")
    
    st.markdown("""
    **Model Used:** 🌲 Random Forest Classifier (with GridSearchCV tuning)  
    - 🎛️ Optimized using `GridSearchCV`  
    - 🔄 Handles both categorical and numeric features using `OneHotEncoding` pipeline  
    - 🎯 Target: `treatment` (whether the person seeks mental health treatment)
    """)

    col1, col2 = st.columns(2)
    with col1:
        self_employed = st.selectbox("💼 Are you self-employed?", ["Yes", "No"])
        family_history = st.selectbox("🧬 Do you have a family history of mental illness?", ["Yes", "No"])
        remote_work = st.selectbox("🏠 Do you work remotely?", ["Yes", "No"])
        care_options = st.selectbox("🏥 Are mental health care options provided at your workplace?", ["Yes", "No", "Not sure"])
        seek_help = st.selectbox("🙋 Does your company encourage seeking help for mental health?", ["Yes", "No", "Don't know"])
        obs_consequence = st.selectbox("👀 Have you seen negative consequences for coworkers discussing mental health?", ["Yes", "No"])
        mental_health_interview = st.selectbox("🗣️ Would you feel comfortable discussing mental health in a job interview?", ["Yes", "No", "Maybe"])
    with col2:
        age = st.slider("🎂 What is your age?", 21, 45)
        work_interfere = st.selectbox("💼 How often does your mental health interfere with your work?", ["Often", "Rarely", "Never", "Sometimes"])
        benefits = st.selectbox("🎁 Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
        wellness_program = st.selectbox("💡 Does your employer offer a wellness program?", ["Yes", "No", "Don't know"])
        leave = st.selectbox("📝 How easy is it for you to take medical leave for mental health?", ["Somewhat easy", "Don't know", "Very difficult", "Somewhat difficult", "Very easy"])
        supervisor = st.selectbox("👔 Do you feel supported by your supervisor regarding mental health?", ["Yes", "No", "Some of them"])
        gender_cleaned = st.selectbox("🚻 What is your gender?", ["Male", "Female", "Other"])

    input_data = pd.DataFrame([{
        "self_employed": self_employed,
        "Age": age,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "remote_work": remote_work,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "leave": leave,
        "obs_consequence": obs_consequence,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "Gender_cleaned": gender_cleaned
    }])

    if st.button("🧠 Predict Treatment Likelihood"):
        prediction = clf_model.predict(input_data)[0]
        st.success(f"🎯 Prediction: {prediction}")

# ======== TAB 2: REGRESSION ========
with tab2:
    st.subheader("📈 Predict Age based on behavioral & workplace features")

    st.markdown("""
    **Model Used:** ⚙️ XGBoost Regressor (GridSearchCV tuned)  
    - 📊 Uses the top 16 features from the original dataset (label encoded).
    """)

    col1, col2 = st.columns(2)
    with col1:
        no_employees = st.selectbox("🏢 How many employees are in your company?", [3, 15, 63, 300, 750, 1200], key="reg_no_employees")
        leave = st.selectbox("📝 How easy is it for you to take leave for mental health?", ["Somewhat easy", "Don't know", "Very difficult", "Somewhat difficult", "Very easy"], key="reg_leave")
        care_options = st.selectbox("🏥 Are mental health care options available at your workplace?", ["Yes", "No", "Not sure"], key="reg_care_options")
        benefits = st.selectbox("🎁 Does your company offer mental health benefits?", ["Yes", "No", "Don't know"], key="reg_benefits")
        coworkers = st.selectbox("🧑‍🤝‍🧑 Are you comfortable discussing mental health with coworkers?", ["Yes", "No", "Some of them"], key="reg_coworkers")
        supervisor = st.selectbox("👔 Do you feel supported by your supervisor?", ["Yes", "No", "Some of them"], key="reg_supervisor")
        mental_health_interview = st.selectbox("🗣️ Would you discuss mental health in a job interview?", ["Yes", "No", "Maybe"], key="reg_mental_health_interview")
        work_interfere = st.selectbox("💼 How often does your mental health interfere with your work?", ["Often", "Rarely", "Never", "Sometimes"], key="reg_work_interfere")

    with col2:
        remote_work = st.selectbox("🏠 Do you work remotely?", ["Yes", "No"], key="reg_remote_work")
        mental_vs_physical = st.selectbox("⚖️ Do you believe mental health is as important as physical health?", ["Yes", "No", "Don't know"], key="reg_mental_vs_physical")
        phys_health_interview = st.selectbox("💬 Would you discuss physical health in a job interview?", ["Yes", "No", "Maybe"], key="reg_phys_health_interview")
        obs_consequence = st.selectbox("👀 Have you witnessed negative consequences for discussing mental health at work?", ["Yes", "No"], key="reg_obs_consequence")
        anonymity = st.selectbox("🕵️ Is your anonymity protected when seeking mental health care?", ["Yes", "No", "Don't know"], key="reg_anonymity")
        wellness_program = st.selectbox("💡 Does your workplace have a wellness program?", ["Yes", "No", "Don't know"], key="reg_wellness_program")
        seek_help = st.selectbox("🙋 Does your company encourage seeking help for mental health issues?", ["Yes", "No", "Don't know"], key="reg_seek_help")
        family_history = st.selectbox("🧬 Do you have a family history of mental illness?", ["Yes", "No"], key="reg_family_history")

    raw_input = {
        "no_employees": no_employees,
        "leave": leave,
        "care_options": care_options,
        "benefits": benefits,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "work_interfere": work_interfere,
        "remote_work": remote_work,
        "mental_vs_physical": mental_vs_physical,
        "phys_health_interview": phys_health_interview,
        "obs_consequence": obs_consequence,
        "anonymity": anonymity,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "family_history": family_history
    }

    try:
        encoded_input = {
            k: label_encoders[k].transform([v])[0] for k, v in raw_input.items()
        }

        if st.button("📈 Predict Age", key="predict_age_button"):
            df_input = pd.DataFrame([encoded_input])
            scaled_input = scaler.transform(df_input)
            prediction = reg_model.predict(scaled_input)[0]
            st.success(f"🎯 Predicted Age: {prediction:.2f} years")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")