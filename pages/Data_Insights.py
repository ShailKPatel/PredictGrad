import streamlit as st
import pandas as pd
import os
from PIL import Image

# Page config
st.set_page_config(page_title="Data Insights", layout="wide", page_icon='📊')
st.title("📊 Data Insights")
st.markdown("""
Explore the underlying dataset powering PredictGrad. Understand how academic, demographic, and administrative features relate to student performance and academic risk.
""")

# Paths
image_dir = os.path.join("EDA", "main_model", "eda_images")
dataset_path = os.path.join("dataset", "student_performance_dataset.csv")

# Load dataset (cached for performance)
@st.cache_data
def load_data():
    return pd.read_csv(dataset_path)

df = load_data()

# --- Summary Stats ---
st.subheader("🔍 Dataset Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Students", df.shape[0])
col2.metric("Total Features", df.shape[1])
col3.metric("Semesters Covered", "1, 2, and 3")

# --- Sample View ---
st.subheader("🧾 Sample Data View")
st.dataframe(df.head(10), use_container_width=True)

# --- EDA Image Panels ---
st.subheader("📌 Feature Relationships and Risk Insights")

def show_image(img_file, caption):
    img_path = os.path.join(image_dir, img_file)
    st.image(Image.open(img_path), caption=caption, use_container_width=True)

# 1. Overall Risk Distribution
show_image("risk_flag_distribution.png", "Distribution of Academic Risk (Sem3_Risk_Flag)")

# 2. Correlation + VIF + Mutual Info
show_image("correlation_heatmap.png", "Feature Correlation Heatmap")
show_image("high_correlation_pairs.png", "Highly Correlated Feature Pairs")
show_image("high_vif_features.png", "Features with High VIF")
show_image("mutual_information_top25.png", "Top 25 Features by Mutual Information")

# 3. Demographics vs Risk
st.markdown("#### 🧑‍🎓 Demographic Influence on Risk")
col4, col5, col6 = st.columns(3)
with col4:
    show_image("gender_vs_risk_flag.png", "Gender vs Risk")
with col5:
    show_image("religion_vs_risk_flag.png", "Religion vs Risk")
with col6:
    show_image("branch_vs_risk_flag.png", "Branch vs Risk")

# 4. Section/Division Influence
st.markdown("#### 🏫 Section/Division vs Risk")
col7, col8, col9 = st.columns(3)
with col7:
    show_image("section1_vs_risk_flag.png", "Section 1 vs Risk")
with col8:
    show_image("section2_vs_risk_flag.png", "Section 2 vs Risk")
with col9:
    show_image("section3_vs_risk_flag.png", "Section 3 vs Risk")

# 5. Academic Performance Distributions
st.subheader("📈 Academic Trends vs Risk")
show_image("sem1_vs_risk_flag.png", "Semester 1 Marks vs Risk")
show_image("sem2_vs_risk_flag.png", "Semester 2 Marks vs Risk")
show_image("pred_sem3_vs_risk_flag.png", "Predicted Sem 3 Performance vs Risk")
show_image("percentile_drop_vs_risk_flag.png", "Percentile Drop vs Risk")

# 6. KDE Views
st.subheader("🧮 Percentile and Score Distributions")
col10, col11 = st.columns(2)
with col10:
    show_image("kde_percentile_drop_vs_risk_flag.png", "KDE: Percentile Drop")
with col11:
    show_image("kde_pred_sem3_vs_risk_flag.png", "KDE: Predicted Sem3 Marks")

col12, col13 = st.columns(2)
with col12:
    show_image("kde_sem1_vs_risk_flag.png", "KDE: Sem1 Marks")
with col13:
    show_image("kde_sem2_vs_risk_flag.png", "KDE: Sem2 Marks")
