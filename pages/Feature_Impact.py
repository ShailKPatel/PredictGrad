import streamlit as st
import os
from PIL import Image

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

# === Utility to display image if it exists ===
def show_image(img_dir, img_file, caption):
    img_path = os.path.join(img_dir, img_file)
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption=caption, use_container_width=True)
    else:
        st.error(f"Missing image: `{img_path}`")

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
st.markdown("## 📊 Subject-Level Model Performance")
subject_base_path = os.path.join("EDA", "SubjectModels", f"{subject_code}_model", "model_performance")

subject_images = [
    ("optimized_ensemble_model_summary.png", "Optimized Ensemble Model Summary"),
    (f"{subject_code}_shap_summary.png", "SHAP Summary"),
    (f"{subject_code}_feature_importance.png", "Feature Importances"),
    (f"{subject_code}_cv_predictions.png", "Cross-Validation Predictions"),
    (f"{subject_code}_actual_vs_predicted.png", "Actual vs Predicted Marks"),
    (f"{subject_code}_prediction_errors.png", "Prediction Errors"),
    (f"{subject_code}_error_distribution.png", "Error Distribution"),
    ("mae_comparison_by_algorithm.png", "MAE Comparison by Algorithm")
]

for file, caption in subject_images:
    show_image(subject_base_path, file, caption)

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

classification_path = os.path.join("EDA", "main_model", "model_performance")
classification_figures = [
    ("stacking_risk_model_summary.png", "Stacking Risk Model Summary"),
    ("actual_vs_predicted_confusion_matrix.png", "Confusion Matrix"),
    ("classification_misclass_distribution.png", "Misclassification Distribution"),
    ("grouped_metrics_per_model.png", "Model Metrics Comparison"),
    ("shap_comparison_base_models.png", "SHAP Comparison Across Base Models")
]

for file, caption in classification_figures:
    show_image(classification_path, file, caption)
