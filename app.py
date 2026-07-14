"""
Titanic Survival Predictor - Streamlit App
--------------------------------------------
Loads the model trained by train_model.py and lets a user enter
passenger details to get a live survival prediction.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# -----------------------------
# Load model + preprocessing objects
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("titanic_model.pkl")
        scaler = joblib.load("scaler.pkl")
        meta = joblib.load("encoders.pkl")
        return model, scaler, meta
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
model, scaler, meta = load_artifacts()

# -----------------------------
# Header
# -----------------------------
st.title("🚢 Titanic Survival Predictor")
st.write(
    "This app uses a machine learning model trained on the classic Titanic dataset "
    "to estimate whether a passenger would have survived, based on their details."
)

with st.expander("About this model"):
    st.write(f"**Model:** {meta['model_name']}")
    st.write(f"**Test accuracy:** {meta['accuracy']:.1%}")
    st.write(
        "Trained on 891 Titanic passengers with engineered features including "
        "title extracted from name, family size, and cabin availability."
    )
    try:
    importance = pd.read_csv("feature_importance.csv", index_col=0)
    importance.columns = ["Importance"]

    st.subheader("Feature Importance")
    st.dataframe(importance)

except FileNotFoundError:
    pass

st.divider()

# -----------------------------
# Input form
# -----------------------------
st.subheader("Enter passenger details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2,
                           help="1 = Upper, 2 = Middle, 3 = Lower")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 28)
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

with col2:
    fare = st.number_input("Fare paid (£)", 0.0, 512.0, 32.0)
    sibsp = st.number_input("Siblings / Spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / Children aboard", 0, 6, 0)
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
    has_cabin = st.checkbox("Had a recorded cabin", value=(pclass == 1))

# -----------------------------
# Build feature vector matching training pipeline
# -----------------------------
embarked_map = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

sex_enc = meta["sex"].transform([sex])[0]
embarked_enc = meta["embarked"].transform([embarked_map[embarked]])[0]
title_enc = meta["title"].transform([title])[0]

row = pd.DataFrame([{
    "Pclass": pclass,
    "Sex_enc": sex_enc,
    "Age": age,
    "Fare": fare,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "HasCabin": int(has_cabin),
    "Embarked_enc": embarked_enc,
    "Title_enc": title_enc,
}])[meta["features"]]

# -----------------------------
# Predict
# -----------------------------
st.divider()

if st.button("Predict survival", type="primary", use_container_width=True):
    X = scaler.transform(row) if meta["uses_scaling"] else row
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        st.success(f"### Likely to survive — {probability:.1%} probability")
    else:
        st.error(f"### Unlikely to survive — {probability:.1%} probability of survival")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "This is a probabilistic estimate from a model trained on historical data — "
        "not a statement of fact about any individual."
    )

st.divider()
st.caption("Built with scikit-learn + Streamlit · Titanic dataset (Kaggle)")
