# PredictGrad: Academic Risk Prediction for Engineering Students

PredictGrad is a machine learning project designed to identify engineering students at academic risk by forecasting their future academic performance. The system predicts Semester 3 core subject marks using data from earlier semesters and flags students likely to experience a significant decline in performance. These insights can support targeted academic interventions and prevent longer-term performance drops.

---

## Problem Statement

This project tackles a two-stage prediction problem:

### 1. Subject-wise Mark Prediction (Regression)

Predict raw theory marks in each core Semester 3 subject using Semester 1 and 2 data:

* **Math-3**
* **Digital Electronics (DE)**
* **Full Stack Development (FSD)**
* **Python**

Each subject is modeled independently to account for unique difficulty levels and performance patterns. The models use features like previous marks, attendance, and engineered aggregates.

### 2. Academic Risk Detection (Classification)

Based on the predicted Semester 3 marks, the system calculates the total and percentile rank of each student. A student is flagged as at risk if their predicted percentile drops by 10 or more points compared to their Semester 2 percentile.

The risk flag (**Sem3_Risk_Flag**) becomes the target variable for a binary classification model. The classifier is trained using both original features and predicted marks, allowing it to learn patterns associated with sudden academic decline.

---

## Dataset

This project uses a structured dataset of academic records collected from a local engineering college, covering three semesters of performance data for undergraduate students across Computer Engineering and related programs.

**Key Details:**

* **Source:** Digitized and anonymized academic records
* **File:** `student_performance_dataset.csv`
* **Total Students:** 905
* **Features:** 56 columns including demographics, theory and practical marks, and attendance
* **Semesters:** 1, 2, and 3
* **Subjects:**
    * Core subjects (e.g., Math, Java, Python) used for prediction and roll assignment
    * Non-core subjects (e.g., Law, Environmental Science) included for completeness but typically excluded from modeling
* **Marks:** All marks (theory and practical) are on a 0–100 scale
* **Attendance:** Available for Semesters 1 and 2 (as percentage); not recorded for Semester 3

**Privacy Measures:**

* All student identifiers have been anonymized.
* Gender and Religion were inferred algorithmically from names and may contain classification noise.

This dataset forms the backbone of both regression and classification pipelines in PredictGrad.

---

## Regression Modeling

To forecast marks in Semester 3 core subjects, independent regression models were built for each subject using prior academic data (Semesters 1 and 2). Each subject underwent a separate hyperparameter search, testing 56 different model pipelines that combined preprocessing, regressors, and tuning strategies.

Although each subject was modeled independently, the same pipeline architecture emerged as the best performer in all four cases.

**Best Model Architecture (All Subjects):**

* **Model:** Voting Regressor (Ridge + Lasso + ElasticNet)
* **Preprocessing:** OneHot Encoding + RobustScaler
* **Validation Strategy:** Repeated 5-Fold Cross-Validation
* **Hyperparameter Tuning:** BayesSearchCV

**Model Comparison by Subject**

| Subject | Models Tried | Best Model Description                                                                     | Cross-Validation MAE | Test MAE |
| :------ | :----------- | :----------------------------------------------------------------------------------------- | :------------------- | :------- |
| Math-3  | 56           | Voting Regressor (Ridge + Lasso + ElasticNet), OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV | 6.3179               | 6.97     |
| DE      | 56           | Voting Regressor (Ridge + Lasso + ElasticNet), OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV | 7.1206               | 7.10     |
| FSD     | 56           | Voting Regressor (Ridge + Lasso + ElasticNet), OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV | 6.5837               | 6.65     |
| Python  | 56           | Voting Regressor (Ridge + Lasso + ElasticNet), OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV | 5.1607               | 5.71     |

These predictions were later used to compute Semester 3 percentile ranks, which formed the basis for academic risk classification.

---

## Risk Classification Model

After predicting individual subject marks, the system computes a student's **Sem3_Percentage** and converts it into a percentile rank (**Sem3_Percentile**). A student is flagged as "at risk" if their predicted **Sem3_Percentile** drops by 10 or more points compared to their **Sem2_Percentile**.

The target variable for classification is:
$$Sem3\_Risk\_Flag = 1 \quad \text{if} \quad Sem3\_Percentile \leq Sem2\_Percentile - 10$$
$$Sem3\_Risk\_Flag = 0 \quad \text{otherwise}$$

To ensure this prediction is made only from Semester 1 and 2 data (i.e., information available before the student enters Semester 3), no predicted Semester 3 marks or future data are used as model input.

**Model Search:**

A total of 52 classification pipelines were evaluated using techniques such as class balancing, stacking, and threshold optimization.

**Best Performing Classifier:**

* **Type:** Stacking Ensemble
* **Base Learners:** CatBoost, BalancedBaggingClassifier (LGBM), ExtraTrees
* **Meta Learner:** Logistic Regression
* **Threshold Tuning:** Optimal threshold identified at 0.49 for F1-recall tradeoff

**Performance Summary:**

| Metric      | Cross-Validation | Test Set |
| :---------- | :--------------- | :------- |
| Accuracy    | 0.6849           | 0.6349   |
| Precision   | 0.3623           | 0.3300   |
| Recall      | 0.7048           | 0.7048   |
| F1-Score    | 0.4735           | 0.5100   |

---

## Streamlit App

The entire PredictGrad pipeline has been deployed using Streamlit to allow users to interact with the models and explore the dataset.

**App Link:**

[App Link: Coming Soon]

**App Pages:**

* **Welcome:** Introduction to the project and its objectives.
* **Predict Risk:** Upload student data or select a student to test the model and predict academic risk.
* **Data Insights:** Explore the dataset, view distributions, and understand key feature summaries.
* **Explain Prediction:** Learn about the model’s predictions with the help of Exploratory Data Analysis (EDA) and Explainable AI (XAI) techniques.
* **Feedback:** Leave comments and suggestions for improvements.
* **Credits:** Acknowledgements for the tools and contributors involved.

---

## Tools & Techniques

* **Modeling:** scikit-learn, CatBoost, LightGBM, ExtraTrees
* **Optimization:** BayesSearchCV
* **EDA/XAI:** SHAP, Seaborn, Matplotlib
* **Deployment:** Streamlit

---

## Contact

For any questions, collaboration, or feedback:

**Shail K Patel**
LinkedIn: [Shail K Patel](https://www.linkedin.com/in/shailkpatel/)