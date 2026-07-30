import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os 

# Load the model
model = joblib.load("xgb_model.pkl")
feature_names = [
    "Calcification", "Envelop.invasion", "Margin", "Age", "Diameter"]

# Streamlit user interface
st.title("Predictor of Skip metastasis in papillary thyroid cancer (Non-menopausal Group)")

# Calcification: categorical selection (1=Yes, 0=No)
Calcification = st.selectbox("Calcification:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Envelop.invasion: categorical selection (1=Yes, 0=No)
Envelop_invasion = st.selectbox("Envelop.invasion:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Margin: categorical selection (0=Clear, 1=Ill-defined)
Margin = st.selectbox("Margin:", options=[0, 1], format_func=lambda x: 'Clear (0)' if x == 0 else 'Ill-defined (1)')

# Age: numerical input (direct number input)
Age = st.number_input("Age (years):", min_value=1, max_value=100, value=45, step=1)

# Diameter: numerical input (unit: mm)
Diameter = st.number_input("Max Diameter (mm):", min_value=0.1, max_value=200.0, value=10.0, step=0.1)

# Process inputs and make predictions
feature_values = [Calcification, Envelop_invasion, Margin, Age, Diameter]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of skip metastasis in papillary thyroid cancer. "
            f"The model predicts that your probability of having skip metastasis in papillary thyroid cancer is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a thyroid surgeon as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of skip metastasis in papillary thyroid cancer. "
            f"The model predicts that your probability of not having skip metastasis in papillary thyroid cancer is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your thyroid health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")