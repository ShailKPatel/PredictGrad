import streamlit as st
import os
import pandas as pd

from EDA.SubjectModels.de_model.models_info import models as de_models

st.set_page_config(page_title="Model Selection", page_icon='⚙️', layout='wide')

st.title("⚙️ Model Selection")

st.markdown("""
Welcome to the Model Selection page of **PredictGrad**.

This page provides an overview of the different models used in the project and their performance metrics.

**Key Insights:**
-  **Model :**
-  **Approach :**
-  **Performance Metrics :**
    - **Mean Absolute Error (MAE)**
            """)   
           
def show_models(subject):
    model_df = pd.DataFrame(eval(f"{subject}_models"))
    for i in range(len(model_df)):
        match i:
            case 0:
                st.markdown("### Linear Regression")
            case 4:
                st.markdown("### Polynomial Regression")
            case 10:
                st.markdown("### Support Vector Regression")
            case 12:
                st.markdown("### Random Forest Regressor")  
            case 20:
                st.markdown("### XGBoost Regressor")
            case 23:
                st.markdown("### LightGBM Regressor")
            case 27:
                st.markdown("### Ridge Regression")
            case 35:
                st.markdown("### Lasso Regression")
            case 39:
                st.markdown("### ElasticNet Regression")
            case 50:
                st.markdown("### Voting Regressor")
            case _:
                pass

        with st.expander(f"{model_df["Model"].iloc[i]} | MAE : {model_df['MAE'].iloc[i]}"):
            st.markdown(f"Approach : {model_df['Approach'].iloc[i]}")
            st.metric("Mean Absolute Error (MAE)", model_df['MAE'].iloc[i])
            st.code(model_df['Code'].iloc[i], language="python")

de_tab, math3_tab, fsd_tab, python_tab = st.tabs(["Digital Electronics (DE)", "Math-3", "Full Stack Development (FSD)", "Python"])

prerequisite_code = """
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# Read the math train data
df = pd.read_csv("../train_dataset.csv")

# Drop the irrelevant, data leak columns
df = df.drop(
    columns=[
        "Student ID",
        "Mentor-1",
        "Mentor-2",
        "Mentor-3",
        "Roll-2",
        "Roll-3",
        "Math-3 Theory",
        "DE Practical",
        "FSD Theory",
        "FSD Practical",
        "Python Theory",
        "Python Practical",
        "Communication Theory",
        "Law Theory",
    ]
)

# columns for Semester 1 core subjects
sem1_columns = [
    "Math-1 Theory",
    "Physics Theory",
    "Java-1 Theory",
    "Software Engineering Theory",
]

# Calculate Semester 1 Percentage as the average of core subject scores
# scores are numerical and out of 100
df["Sem 1 Percentage"] = df[sem1_columns].mean(axis=1).round(2)

# columns for Semester 2 core subjects
sem2_columns = [
    "Math-2 Theory",
    "Data Structures using Java Theory",
    "DBMS Theory",
    "Fundamental of Electronics and Electrical Theory",
    "Java-2 Theory",
]

# Calculate Semester 2 Percentage as the average of core subject scores
# scores are numerical and out of 100
df["Sem 2 Percentage"] = df[sem2_columns].mean(axis=1).round(2)


# Rename columns Div-1, Div-2, Div-3 to Section-1, Section-2, Section-3
df = df.rename(
    columns={"Div-1": "Section-1", "Div-2": "Section-2", "Div-3": "Section-3"}
)

# Transform values in Section-1, Section-2, Section-3 to keep only the first character
# Thus we get Only Department
for section in ["Section-1", "Section-2", "Section-3"]:
    df[section] = df[section].str[0]
"""

with de_tab:
    st.markdown("#### Prerequisite Code for DE Models (Used for all DE Models)")
    st.code(prerequisite_code, language="python")
    st.markdown("---")

    st.markdown("## Digital Electronics (DE) Models")
    show_models("de")

with math3_tab:
    st.markdown("### Math-3 Model")

with fsd_tab:
    st.markdown("### Full Stack Development (FSD) Model")

with python_tab:
    st.markdown("### Python Model")
