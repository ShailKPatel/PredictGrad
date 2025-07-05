import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import warnings
from io import BytesIO
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

# Add the correct absolute paths to model directories
sys.path.append(os.path.abspath("model"))

# Now import directly from the files in model/
from de_handler import DEModelHandler
from fsd_handler import FSDModelHandler
from math3_handler import Math3ModelHandler
from python_handler import PythonModelHandler
from main_model_handler import RiskModelHandler

REQUIRED_COLUMNS = [
    "Student ID",
    "Gender",
    "Religion",
    "Branch",
    "Div-1",
    "Div-2",
    "Div-3",
    "Roll-1",
    "Roll-2",
    "Roll-3",
    "Mentor-1",
    "Mentor-2",
    "Mentor-3",
    "Math-1 Theory",
    "Physics Theory",
    "Physics Practical",
    "Java-1 Theory",
    "Java-1 Practical",
    "Software Engineering Theory",
    "Software Engineering Practical",
    "Environmental Science Theory",
    "IOT Workshop Practical",
    "Computer Workshop Practical",
    "Math-2 Theory",
    "Data Structures using Java Theory",
    "Data Structures using Java Practical",
    "DBMS Theory",
    "DBMS Practical",
    "Fundamental of Electronics and Electrical Theory",
    "Fundamental of Electronics and Electrical Practical",
    "Java-2 Theory",
    "Java-2 Practical",
    "Math-1 Attendance",
    "Physics Attendance",
    "Java-1 Attendance",
    "Software Engineering Attendance",
    "Environmental Science Attendance",
    "IOT Workshop Attendance",
    "Math-2 Attendance",
    "Data Structures using Java Attendance",
    "DBMS Attendance",
    "Fundamental of Electronics and Electrical Attendance",
    "Java-2 Attendance",
]

st.set_page_config(page_title="Predict Academic Risk", layout="wide")
st.title("🎓 Predict Academic Risk of Engineering Students")

st.markdown("""
Upload a CSV file with Semester 1 & 2 student data to predict:
1. Subject-wise Semester 3 marks (Math-3, DE, FSD, Python)
2. Whether the student is **at risk** due to percentile drop

---
#### ⚠️ Input Requirements:
- File must contain **exactly 45 required columns** listed below
- Types must match: theory/attendance/practical columns must be numeric
- If format doesn't match, an error will be shown
""")

with st.expander("📑 View Required Columns"):
    st.code("\n".join(REQUIRED_COLUMNS), language="text")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Check for required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.error(f"CSV is missing required columns: {missing_cols}")
            st.stop()

        # Check if the file has at least 5 rows
        if df.shape[0] < 5:
            st.error(f"CSV must contain at least 5 columns. Found only {df.shape[0]}.")
            st.stop()

        # Check types of numerical columns
        numeric_cols = [
            col
            for col in REQUIRED_COLUMNS
            if "Theory" in col
            or "Attendance" in col
            or "Practical" in col
            or "Roll" in col
        ]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric")

        dm = df.copy()
        dm.index = range(1, len(df) + 1)
        dm.index.name = ""  # optional: remove the "index" label

        st.dataframe(dm, use_container_width=True)
        
        # Clean data and run handlers
        df_clean = df.drop(
            columns=[
                "Mentor-1",
                "Mentor-2",
                "Mentor-3",  # Mentor-3 might not exist
                "Roll-2",
                "Roll-3",  # These might not exist
                "Math-3 Theory",
                "DE Theory",
                "DE Practical",
                "FSD Theory",
                "FSD Practical",
                "Python Theory",
                "Python Practical",
                "Communication Theory",
                "Law Theory",
            ],
            errors="ignore",
        )
        df_clean["Sem 1 Percentage"] = (
            df[
                [
                    "Math-1 Theory",
                    "Physics Theory",
                    "Java-1 Theory",
                    "Software Engineering Theory",
                ]
            ]
            .mean(axis=1)
            .round(2)
        )

        df_clean["Sem 2 Percentage"] = (
            df[
                [
                    "Math-2 Theory",
                    "Data Structures using Java Theory",
                    "DBMS Theory",
                    "Fundamental of Electronics and Electrical Theory",
                    "Java-2 Theory",
                ]
            ]
            .mean(axis=1)
            .round(2)
        )

        df_clean = df_clean.rename(
            columns={"Div-1": "Section-1", "Div-2": "Section-2", "Div-3": "Section-3"}
        )
        for section in ["Section-1", "Section-2", "Section-3"]:
            df_clean[section] = df_clean[section].astype(str).str[0]

        # Subject predictions
        handlers = {
            "Predicted DE Theory": (DEModelHandler, "model/de_model.joblib"),
            "Predicted FSD Theory": (FSDModelHandler, "model/fsd_model.joblib"),
            "Predicted Math-3 Theory": (Math3ModelHandler, "model/math3_model.joblib"),
            "Predicted Python Theory": (
                PythonModelHandler,
                "model/python_model.joblib",
            ),
        }

        for col_name, (handler_cls, path) in handlers.items():
            handler = handler_cls()
            result = handler.predict_from_model(df, model_path=path, return_type="df")
            df_clean[col_name] = result[col_name]

        sem3_cols = list(handlers.keys())
        df_clean["Predicted Sem 3 Percentage"] = (
            df_clean[sem3_cols].mean(axis=1).round(2)
        )

        model_handler = RiskModelHandler(
            model_path="../model/stacking_risk_model.joblib"
        )
        df_clean["Risk Prediction"] = model_handler.predict(df)

        st.success("✅ Prediction Successful")
        df_display = df_clean[
            [
                "Student ID",
                "Predicted Math-3 Theory",
                "Predicted DE Theory",
                "Predicted FSD Theory",
                "Predicted Python Theory",
                "Predicted Sem 3 Percentage",
                "Risk Prediction",
            ]
        ].reset_index(drop=True)

        # Set the DataFrame index to start from 1
        df_display.index = range(1, len(df_display) + 1)
        df_display.index.name = ""  # optional: remove the "index" label

        st.dataframe(df_display, use_container_width=True)

        # Download button
        csv_output = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prediction CSV",
            data=csv_output,
            file_name="predicted_student_risk.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(
            "❌ Invalid CSV Format or Data Type Error: Please ensure your file matches the documented dataset format."
        )
        st.exception(e)
