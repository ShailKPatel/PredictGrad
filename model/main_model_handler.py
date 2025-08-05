import os
import sys
import pandas as pd
import joblib

# Get the directory where this file resides
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Add model directory to path (in case handlers are imported locally)
sys.path.append(MODEL_DIR)

from de_handler import DEModelHandler
from fsd_handler import FSDModelHandler
from math3_handler import Math3ModelHandler
from python_handler import PythonModelHandler


class RiskModelHandler:
    def __init__(self, model_path="risk_model.joblib"):
        # Resolve the full model path
        model_path = os.path.join(MODEL_DIR, model_path)

        # Load the joblib file and extract all components
        model_data = joblib.load(model_path)
        self.model = model_data.get("model")
        self.selected_features = model_data.get("features")
        self.threshold = model_data.get("threshold")

        if (
            self.model is None
            or self.selected_features is None
            or self.threshold is None
        ):
            raise ValueError(
                "Joblib file must contain 'model', 'features', and 'threshold'."
            )

        # Initialize subject-specific model handlers
        self.de = DEModelHandler()
        self.fsd = FSDModelHandler()
        self.math3 = Math3ModelHandler()
        self.python = PythonModelHandler()

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        # Generate predictions for subject-specific models
        df["Predicted DE Theory"] = self.de.predict_from_model(
            df, os.path.join(MODEL_DIR, "de_model.joblib"), "df"
        )["Predicted DE Theory"]

        df["Predicted FSD Theory"] = self.fsd.predict_from_model(
            df, os.path.join(MODEL_DIR, "fsd_model.joblib"), "df"
        )["Predicted FSD Theory"]

        df["Predicted Math-3 Theory"] = self.math3.predict_from_model(
            df, os.path.join(MODEL_DIR, "math3_model.joblib"), "df"
        )["Predicted Math-3 Theory"]

        df["Predicted Python Theory"] = self.python.predict_from_model(
            df, os.path.join(MODEL_DIR, "python_model.joblib"), "df"
        )["Predicted Python Theory"]

        # Derived features
        df["Predicted Sem 3 Percentage"] = (
            df[
                [
                    "Predicted Math-3 Theory",
                    "Predicted DE Theory",
                    "Predicted FSD Theory",
                    "Predicted Python Theory",
                ]
            ]
            .mean(axis=1)
            .round(2)
        )

        df["Sem 2 Percentage"] = df[
            [
                "Math-2 Theory",
                "Data Structures using Java Theory",
                "DBMS Theory",
                "Fundamental of Electronics and Electrical Theory",
                "Java-2 Theory",
            ]
        ].mean(axis=1)
        df["Sem 2 Percentile"] = df["Sem 2 Percentage"].rank(pct=True) * 100

        df["Predicted Sem 3 Percentile"] = (
            df["Predicted Sem 3 Percentage"].rank(pct=True) * 100
        )
        df["Predicted Percentile Drop"] = (
            df["Sem 2 Percentile"] - df["Predicted Sem 3 Percentile"]
        ).round(2)
        df["Predicted Risk Flag"] = (df["Predicted Percentile Drop"] > 10).astype(bool)

        df["Sem 1 Percentage"] = (
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

        df.rename(
            columns={"Div-1": "Section-1", "Div-2": "Section-2", "Div-3": "Section-3"},
            inplace=True,
        )
        for sec in ["Section-1", "Section-2", "Section-3"]:
            df[sec] = df[sec].str[0]

        df["Sem 1 Percentile"] = df["Sem 1 Percentage"].rank(pct=True) * 100
        df["Sem 2 Percentile"] = df["Sem 2 Percentage"].rank(pct=True) * 100

        # Drop raw sem3 cols and unused
        df.drop(
            columns=[
                "Student ID",
                "Mentor-1",
                "Mentor-2",
                "Mentor-3",
                "Roll-2",
                "Roll-3",
                "Math-3 Theory",
                "DE Theory",
                "DE Practical",
                "FSD Theory",
                "FSD Practical",
                "Python Theory",
                "Python Practical",
                "Communication Theory",
                "Law Theory",
                "Sem 3 Percentage",
                "Sem 3 Percentile",
                "Percentile Drop",
                "Risk Flag",
            ],
            inplace=True,
            errors="ignore",
        )

        return df

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align input df with training-time columns"""
        df = df.copy()
        for col in self.selected_features:
            if col not in df.columns:
                df[col] = 0
        return df[self.selected_features]

    def predict(self, df: pd.DataFrame):
        df = self._feature_engineer(df)
        df = pd.get_dummies(df)
        df = self._align_columns(df)
        return self.model.predict(df)

    def predict_proba(self, df: pd.DataFrame):
        df = self._feature_engineer(df)
        df = pd.get_dummies(df)
        df = self._align_columns(df)
        return self.model.predict_proba(df)
  