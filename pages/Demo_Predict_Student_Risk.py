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

st.set_page_config(page_title="Demo Predict Academic Risk", layout="wide")
st.title("🎓 Predict Academic Risk of Engineering Students")

st.markdown("""
This demo uses a **pre-selected sample dataset** to showcase predictions:
1. Subject-wise Semester 3 marks (Math-3, DE, FSD, Python)
2. Whether the student is **at risk** due to a percentile drop

---
#### ⚠️ Demo Notes:
- This version does **not** support file uploads.
- Data shown is for **illustration purposes only**, using predefined student records.
- In production, users can upload a properly formatted CSV.

#### 📌 Required Input Format:
- File must contain **exactly 45 columns** listed below
- All theory/practical/attendance/roll columns must be numeric
- Missing or malformed columns will raise an error
""")

with st.expander("📑 View Required Columns"):
    st.code("\n".join(REQUIRED_COLUMNS), language="text")


if True:
    try:
        data = [
            [71, 'M', 'Hindu', 'CE', 'D2', 'D1', 'A1', 46, 21, 28, 'MMS', 'PDB', 'MVP', 82, 80, 81, 76, 85, 72, 86, 79, 86, 87, 66, 71, 90, 75, 82, 81, 90, 79, 89, 88.0, 81.0, 90.0, 86.0, 85.0, 100.0, 82.81, 83.61, 80.65, 83.33, 77.27],
            [458, 'M', 'Hindu', 'CS&IT', 'A2', 'A6', 'D6', 41, 150, 184, 'PBS', 'HDS', 'MVK', 47, 48, 51, 33, 69, 39, 55, 42, 76, 83, 48, 50, 63, 36, 54, 45, 40, 53, 68, 93.1, 88.71, 94.85, 88.1, 66.67, 100.0, 84.06, 78.46, 74.19, 76.09, 77.97],
            [219, 'M', 'Hindu', 'IT', 'D5', 'D9', 'C3', 148, 263, 102, 'MMS', 'IAM', 'SDS', 63, 44, 73, 41, 75, 64, 68, 56, 81, 96, 49, 46, 63, 60, 64, 69, 82, 62, 76, 92.0, 86.0, 91.0, 79.0, 79.0, 100.0, 88.06, 88.57, 84.48, 98.00, 92.98],
            [251, 'M', 'Hindu', 'IT', 'D5', 'D8', 'C3', 122, 233, 73, 'DPP', 'VHP', 'BAP', 70, 53, 78, 50, 87, 52, 61, 46, 58, 88, 68, 66, 68, 43, 62, 63, 69, 61, 72, 92.0, 94.0, 94.0, 90.0, 93.0, 100.0, 100.0, 96.83, 98.25, 100.00, 98.39],
            [40, 'M', 'Hindu', 'CE', 'D3', 'D9', 'A5', 82, 261, 163, 'NSD', 'ASA', 'DPB', 64, 57, 86, 37, 64, 55, 70, 57, 80, 78, 79, 59, 73, 42, 49, 51, 71, 43, 73, 92.0, 95.0, 94.0, 93.0, 93.0, 100.0, 97.01, 97.14, 100.00, 96.00, 100.00]
        ]

        columns = [
            "Student ID", "Gender", "Religion", "Branch", "Div-1", "Div-2", "Div-3",
            "Roll-1", "Roll-2", "Roll-3", "Mentor-1", "Mentor-2", "Mentor-3",
            "Math-1 Theory", "Physics Theory", "Physics Practical", "Java-1 Theory",
            "Java-1 Practical", "Software Engineering Theory", "Software Engineering Practical",
            "Environmental Science Theory", "IOT Workshop Practical", "Computer Workshop Practical",
            "Math-2 Theory", "Data Structures using Java Theory", "Data Structures using Java Practical",
            "DBMS Theory", "DBMS Practical", "Fundamental of Electronics and Electrical Theory",
            "Fundamental of Electronics and Electrical Practical", "Java-2 Theory", "Java-2 Practical",
            "Math-1 Attendance", "Physics Attendance", "Java-1 Attendance", "Software Engineering Attendance",
            "Environmental Science Attendance", "IOT Workshop Attendance", "Math-2 Attendance",
            "Data Structures using Java Attendance", "DBMS Attendance",
            "Fundamental of Electronics and Electrical Attendance", "Java-2 Attendance"
        ]


        df = pd.DataFrame(data, columns=columns)

        st.markdown("""
            - This is a sample dataset for demonstration purposes.
            - You can upload your own CSV file with the required columns to predict student risk on "Predict Student Risk" page.
            - Uploading a file is disabled in this demo variant.

            """)
        
        dm = df.copy()
        dm.index = range(1, len(df) + 1)
        dm.index.name = ""  # optional: remove the "index" label

        st.dataframe(dm, use_container_width=True)
        
        

        # Check for required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.error(f"CSV is missing required columns: {missing_cols}")
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
