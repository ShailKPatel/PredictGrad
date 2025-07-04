import streamlit as st
import os

# Set page config
st.set_page_config(page_title="🔍 Feature Impact", layout="wide")

st.title("🔍 Feature Impact Analysis")

st.markdown("""
Welcome to the Feature Impact page of **PredictGrad**.

This analysis explains how different academic and demographic features influence predictions across two stages:

1. 🎯 **Subject Mark Prediction (Regression):**  
   Predict Semester 3 marks for:
   - Math-3
   - Digital Electronics (DE)
   - Full Stack Development (FSD)
   - Python

2. ⚠️ **Academic Risk Flagging (Classification):**  
   A student is flagged as "at risk" if their predicted Semester 3 percentile drops by **10 or more points** compared to their Semester 2 percentile.

Understanding which features drive these predictions helps uncover academic risks early and enables targeted intervention.
""")

# === Subject selector ===
subject_map = {
    "Digital Electronics (DE)": "de",
    "Math-3": "math3",
    "Full Stack Development (FSD)": "fsd",
    "Python": "python"
}
selected_subject = st.selectbox("📘 Select Subject for Feature Analysis", list(subject_map.keys()))
subject_code = subject_map[selected_subject]

# === Subject-level visualizations ===
base_path = f"EDA/SubjectModels/{subject_code}_model/model_performance"

subject_images = [
    "optimized_ensemble_model_summary.png",
    f"{subject_code}_shap_summary.png",
    f"{subject_code}_feature_importance.png",
    f"{subject_code}_cv_predictions.png",
    f"{subject_code}_actual_vs_predicted.png",
    f"{subject_code}_prediction_errors.png",
    f"{subject_code}_error_distribution.png",
    "mae_comparison_by_algorithm.png"
]

st.markdown("## 📊 Subject-Level Model Performance")

for img_file in subject_images:
    full_path = os.path.join(base_path, img_file)
    if os.path.exists(full_path):
        st.image(full_path, width=900)
    else:
        st.warning(f"Missing visualization: `{img_file}`")

# === Academic Risk Classification Visuals ===
st.markdown("---")
st.subheader("📉 Academic Risk Classification Overview")

st.markdown("""
Explore key insights from the risk classification pipeline.  
These visualizations highlight model architecture, performance distribution, and feature impact:

- 🧩 Model Architecture & Summary
- 📊 Confusion Matrix & Misclassification
- 📈 Metric Comparisons
- 🔍 SHAP Feature Contributions
""")

classification_path = "EDA/Main Model/model_performance"
classification_figures = [
    "stacking_risk_model_summary.png",                  # Model pipeline and final stacking configuration
    "actual_vs_predicted_confusion_matrix.png",         # Visual summary of classification outcomes
    "classification_misclass_distribution.png",         # Breakdown of misclassified vs. correctly predicted
    "grouped_metrics_per_model.png",                    # Model selection performance comparison
    "shap_comparison_base_models.png"                   # SHAP feature importance across base learners
]

for img_file in classification_figures:
    full_path = os.path.join(classification_path, img_file)
    if os.path.exists(full_path):
        st.image(full_path, width=900)
    else:
        st.warning(f"Missing visualization: `{img_file}`")

