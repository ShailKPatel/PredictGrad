import streamlit as st
import os
from PIL import Image

# Set page config
st.set_page_config(page_title="Model Analysis", page_icon='🔍', layout='centered')

st.title("🔍 Model Analysis")

st.markdown("""
Welcome to the Model Analysis page of **PredictGrad**.

This analysis explains how different academic and demographic features influence predictions across two stages:

1.  **Subject Mark Prediction (Regression):**  
   Predict Semester 3 marks for:
   - Math-3
   - Digital Electronics (DE)
   - Full Stack Development (FSD)
   - Python

2.  **Academic Risk Flagging (Classification):**  
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

subject_tab, risk_tab = st.tabs(["Subject Models", "Main Risk Model"])

with subject_tab:
    # === Subject selector ===
    st.markdown("## Subject-Level Model Performance")
    st.markdown("""
    Select a subject to view its model performance. This includes visualizations for:
    - Optimized Ensemble Model Summary
    - SHAP Summary
    - Feature Importances
    - Cross-Validation Predictions
    - Actual vs Predicted Marks
    - Prediction Errors
    - Error Distribution
    - MAE Comparison by Algorithm   
    """)
    
    subject_code = "de"  # Default to Digital Electronics for initial load
        
    de_tab, math3_tab, fsd_tab, python_tab = st.tabs(["Digital Electronics (DE)", "Math-3", "Full Stack Development (FSD)", "Python"], width=550)
        
    with de_tab:
        st.markdown("### Digital Electronics (DE) Model Performance")
        subject_code = "de"
        
            # === Subject-level visualizations ===
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
        
    with math3_tab:
        st.markdown("### Math-3 Model Performance")
        subject_code = "math3"
        
            # === Subject-level visualizations ===
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
        
    with fsd_tab:
        st.markdown("### Full Stack Development (FSD) Model Performance")
        subject_code = "fsd"
        
            # === Subject-level visualizations ===
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
        
    with python_tab:
        st.markdown("### Python Model Performance")
        subject_code = "python"
        
            # === Subject-level visualizations ===
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
        
with risk_tab:
    st.markdown("## 📉 Academic Risk Classification Overview")

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

