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
# 主标题修改为论文题目
st.title("Female Central Lymph Node Metastasis Prediction Model for Papillary Thyroid Carcinoma")

# 分组选择，删掉括号多余注释
menopause_status = st.radio(
    "Select Patient Menopausal Status",
    options=[0, 1],
    format_func=lambda x: "Premenopausal" if x == 0 else "Postmenopausal"
)

# 绑定当前模型、特征名，精简副标题
if menopause_status == 0:
    cur_model = model_pre
    cur_feature_names = feature_names_pre
    st.subheader("Current Group: Premenopausal Women")
else:
    cur_model = model_post
    cur_feature_names = feature_names_post
    st.subheader("Current Group: Postmenopausal Women")

# ---------------------- 3. 动态生成输入框（统一精简命名） ----------------------
# 两组共用基础特征
Calcification = st.selectbox("Calcification", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Capsular_Invasion = st.selectbox("Capsular Invasion", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Margin = st.selectbox("Margin", options=[0, 1], format_func=lambda x: 'Clear (0)' if x == 0 else 'Ill-defined (1)')
Diameter = st.number_input("Max Diameter (mm)", min_value=0.1, max_value=200.0, value=10.0, step=0.1)

# 分分组展示独有输入项
Age = None
SII = None
if menopause_status == 0:
    # 非绝经：显示年龄输入
    Age = st.number_input("Patient Age (years)", min_value=1, max_value=100, value=38, step=1)
else:
    # 绝经：显示SII输入，无年龄
    SII = st.number_input("Systemic Immune-Inflammation Index (SII)", min_value=0.0, max_value=5000.0, value=650.0, step=0.1)

# 组装对应分组特征数组（严格匹配训练顺序）
if menopause_status == 0:
    feature_vals = [Calcification, Capsular_Invasion, Margin, Age, Diameter]
else:
    feature_vals = [Calcification, Capsular_Invasion, Margin, Diameter, SII]
input_array = np.array([feature_vals])

# ---------------------- 4. 预测 + SHAP力图绘图 + 文案修改（CLNM替换skip metastasis） ----------------------
if st.button("Start Prediction & Generate SHAP Force Plot"):
    # 模型预测
    pred_class = cur_model.predict(input_array)[0]
    pred_proba = cur_model.predict_proba(input_array)[0]
    risk_percent = pred_proba[pred_class] * 100

    st.divider()
    st.subheader("Prediction Results")
    st.write(f"**Predicted Outcome Class:** {pred_class}")
    st.write(f"**Prediction Probability Array:** {np.round(pred_proba, 3)}")

    # 临床建议文本：全部改为中央区淋巴结转移CLNM
    group_text = "premenopausal" if menopause_status == 0 else "postmenopausal"
    if pred_class == 1:
        advice = (
            f"Based on the {group_text} subgroup XGBoost model, this patient has a high risk of central lymph node metastasis (CLNM) in papillary thyroid carcinoma. "
            f"The predicted probability of central lymph node metastasis is {risk_percent:.1f}%. "
            "This result is only a predictive reference. It is recommended to consult a thyroid surgeon "
            "for comprehensive ultrasound evaluation and personalized surgical planning as soon as possible."
        )
    else:
        advice = (
            f"Based on the {group_text} subgroup XGBoost model, this patient has a low risk of central lymph node metastasis (CLNM) in papillary thyroid carcinoma. "
            f"The predicted probability of no central lymph node metastasis is {risk_percent:.1f}%. "
            "Regular thyroid ultrasound follow-up is still recommended. Seek medical consultation promptly "
            "if nodule enlargement or new malignant signs appear."
        )
    st.info(advice)

    # SHAP绘图，自动适配当前分组变量
    explainer = shap.TreeExplainer(cur_model)
    input_df = pd.DataFrame([feature_vals], columns=cur_feature_names)
    shap_vals = explainer.shap_values(input_df)

    plt.figure()
    shap.force_plot(explainer.expected_value, shap_vals[0], input_df, matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.close()  # 释放画布，避免重复绘图缓存异常

    st.divider()
    st.subheader("Individual SHAP Force Plot")
    st.image("shap_force_plot.png")
