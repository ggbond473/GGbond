import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# ---------------------- 1. 加载两组独立模型 ----------------------
# 非绝经模型（5特征：含Age，无SII）
model_pre = joblib.load("xgb_premenopausal.pkl")
# 绝经模型（5特征：含SII，无Age）
model_post = joblib.load("xgb_postmenopausal.pkl")

# 两组独立特征名（严格匹配训练&图中顺序）
feature_names_pre = [
    "Calcification", "Capsular Invasion", "Margin", "Age", "Max Diameter"
]
feature_names_post = [
    "Calcification", "Capsular Invasion", "Margin", "Max Diameter", "SII"
]

# ---------------------- 2. 页面基础设置 & 绝经状态选择 ----------------------
st.set_page_config(page_title="Thyroid CLNM Predictor")
# 论文正式主标题
st.title("Female Central Lymph Node Metastasis Prediction Model for Papillary Thyroid Carcinoma")

# 分组单选，删除多余括号注释
menopause_status = st.radio(
    "Select Patient Menopausal Status",
    options=[0, 1],
    format_func=lambda x: "Premenopausal" if x == 0 else "Postmenopausal"
)

# 绑定当前模型与特征列表
if menopause_status == 0:
    cur_model = model_pre
    cur_feature_names = feature_names_pre
    st.subheader("Current Group: Premenopausal Women")
else:
    cur_model = model_post
    cur_feature_names = feature_names_post
    st.subheader("Current Group: Postmenopausal Women")

# ---------------------- 3. 输入框定义（去除(0)(1)后缀） ----------------------
Calcification = st.selectbox("Calcification", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Capsular_Invasion = st.selectbox("Capsular Invasion", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Margin = st.selectbox("Margin", options=[0, 1], format_func=lambda x: 'Smooth' if x == 0 else 'Non-smooth')
Diameter = st.number_input("Max Diameter (mm)", min_value=0.1, max_value=200.0, value=10.0, step=0.1)

# 分组独有输入项
Age = None
SII = None
if menopause_status == 0:
    Age = st.number_input("Patient Age (years)", min_value=1, max_value=100, value=38, step=1)
else:
    SII = st.number_input("Systemic Immune-Inflammation Index (SII)", min_value=0.0, max_value=5000.0, value=650.0, step=0.1)

# 组装特征数组
if menopause_status == 0:
    feature_vals = [Calcification, Capsular_Invasion, Margin, Age, Diameter]
else:
    feature_vals = [Calcification, Capsular_Invasion, Margin, Diameter, SII]
input_array = np.array([feature_vals])

# ---------------------- 4. 预测、文案、SHAP绘图 ----------------------
if st.button("Start Prediction & Generate SHAP Force Plot"):
    pred_proba = cur_model.predict_proba(input_array)[0]
    pred_class = cur_model.predict(input_array)[0]
    risk_percent = pred_proba[1] * 100

    st.divider()
    st.subheader("Prediction Results")
    st.write(f"**Predicted Category:** {'CLNM Positive' if pred_class == 1 else 'CLNM Negative'}")
    st.write(f"**Probability of Central Lymph Node Metastasis:** {risk_percent:.2f}%")

    group_text = "premenopausal" if menopause_status == 0 else "postmenopausal"
    if pred_class == 1:
        advice = (
            f"Based on the {group_text} subgroup XGBoost model, this patient has a high risk of central lymph node metastasis (CLNM) in papillary thyroid carcinoma. "
            "This result is only a predictive reference. It is recommended to consult a thyroid surgeon "
            "for comprehensive ultrasound evaluation and personalized surgical planning as soon as possible."
        )
    else:
        advice = (
            f"Based on the {group_text} subgroup XGBoost model, this patient has a low risk of central lymph node metastasis (CLNM) in papillary thyroid carcinoma. "
            "Regular thyroid ultrasound follow-up is still recommended. Seek medical consultation promptly "
            "if nodule enlargement or new malignant signs appear."
        )
    st.info(advice)

    # SHAP绘图
    explainer = shap.TreeExplainer(cur_model)
    input_df = pd.DataFrame([feature_vals], columns=cur_feature_names)
    shap_vals = explainer.shap_values(input_df)

    plt.figure()
    shap.force_plot(explainer.expected_value, shap_vals[0], input_df, matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.close()

    st.divider()
    st.subheader("Individual SHAP Force Plot")
    st.image("shap_force_plot.png")
