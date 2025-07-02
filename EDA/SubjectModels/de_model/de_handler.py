import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
import joblib
import os

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = None
        self.categorical_cols = None
        self.numeric_cols = None

    def _preprocess(self, df):
        EXPECTED_COLS = [
            "Gender", "Religion", "Branch", "Div-1", "Div-2", "Div-3", "Roll-1",
            "Math-1 Theory", "Physics Theory", "Physics Practical",
            "Java-1 Theory", "Java-1 Practical", "Software Engineering Theory", "Software Engineering Practical",
            "Environmental Science Theory", "IOT Workshop Practical", "Computer Workshop Practical",
            "Math-2 Theory", "Data Structures using Java Theory", "Data Structures using Java Practical",
            "DBMS Theory", "DBMS Practical", "Fundamental of Electronics and Electrical Theory",
            "Fundamental of Electronics and Electrical Practical", "Java-2 Theory", "Java-2 Practical",
            "Math-1 Attendance", "Physics Attendance", "Java-1 Attendance", "Software Engineering Attendance",
            "Environmental Science Attendance", "IOT Workshop Attendance", "Math-2 Attendance",
            "Data Structures using Java Attendance", "DBMS Attendance",
            "Fundamental of Electronics and Electrical Attendance", "Java-2 Attendance"
        ]
        
        df = df[[col for col in EXPECTED_COLS if col in df.columns]].copy()

        sem1 = ["Math-1 Theory", "Physics Theory", "Java-1 Theory", "Software Engineering Theory"]
        sem2 = [
            "Math-2 Theory", "Data Structures using Java Theory", "DBMS Theory",
            "Fundamental of Electronics and Electrical Theory", "Java-2 Theory"
        ]
        df["Sem 1 Percentage"] = df[sem1].mean(axis=1).round(2)
        df["Sem 2 Percentage"] = df[sem2].mean(axis=1).round(2)

        df = df.rename(columns={"Div-1": "Section-1", "Div-2": "Section-2", "Div-3": "Section-3"})
        for sec in ["Section-1", "Section-2", "Section-3"]:
            df[sec] = df[sec].astype(str).str[0]

        return df

    def fit(self, X, y=None):
        df = self._preprocess(X)

        self.categorical_cols = [
            "Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"
        ]
        self.numeric_cols = [col for col in df.columns if col not in self.categorical_cols]

        self.pipeline = ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), self.categorical_cols),
            ("num", RobustScaler(), self.numeric_cols),
        ])
        self.pipeline.fit(df)
        return self

    def transform(self, X):
        df = self._preprocess(X)
        return self.pipeline.transform(df)

class DEModelHandler:
    @staticmethod
    def predict_from_model(raw_df, model_path="de_model.joblib", return_type="df"):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at: {model_path}")

        model = joblib.load(model_path)

        if not hasattr(model, "predict"):
            raise ValueError("Loaded object is not a valid sklearn predictor.")

        predictions = model.predict(raw_df)

        if return_type == "df":
            return pd.DataFrame(predictions, columns=["Predicted DE Theory"])
        return predictions
