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
[**File:** `student_performance_dataset.csv`](https://github.com/ShailKPatel/PredictGrad/blob/main/dataset/student_performance_dataset.csv)* 
**Total Students:** 905
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

A total of 50 classification pipelines were evaluated using techniques such as class balancing, stacking, and threshold optimization.

**Best Performing Classifier:**

* **Type:** StackingClassifier
* **Base Learners:** CatBoost, BalancedBaggingClassifier (LGBM), ExtraTrees
* **Meta Learner:** Logistic Regression (`predict_proba` passthrough)  
* **Threshold Tuning:** Optimal threshold identified at 0.500 for F1-recall tradeoff
* **Preprocessing & Feature Selection:**
    OneHotEncoder + Mutual Information (Top 40)  
    Boruta selection from Top 40

* **Cross-Validation & Threshold Tuning:**
    5-Fold Stratified Cross-Validation  
    F1 threshold sweep from 0.2 to 0.7

**Performance Summary:**

| Metric      | Cross-Validation | Test Set |
| :---------- | :--------------- | :------- |
| Accuracy    | 0.6878           | 0.6740   |
| Precision   | 0.3611           | 0.3286   |
| Recall      | 0.7123           | 0.6571   |
| F1-Score    | 0.4793           | 0.4381   |

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

## File Structure

```text
├── EDA/
│   ├── SubjectModels/
│   │   ├── de_model/
│   │   │   ├── de_eda.ipynb
│   │   │   ├── de_handler.py
│   │   │   ├── de_model.joblib
│   │   │   ├── de_model_test_eval.ipynb
│   │   │   ├── model_performance/
│   │   │   └── model_results_log.csv
│   │   ├── fsd_model/
│   │   │   ├── fsd_eda.ipynb
│   │   │   ├── fsd_handler.py
│   │   │   ├── fsd_model.joblib
│   │   │   ├── fsd_model_test_eval.ipynb
│   │   │   ├── model_performance/
│   │   │   └── model_results_log.csv
│   │   ├── math3_model/
│   │   │   ├── math3_handler.py
│   │   │   ├── math3_model.joblib
│   │   │   ├── math3_model_test_eval.ipynb
│   │   │   ├── math_eda.ipynb
│   │   │   ├── model_performance/
│   │   │   └── model_results_log.csv
│   │   ├── python_model/
│   │   │   ├── model_performance/
│   │   │   ├── model_results_log.csv
│   │   │   ├── python_eda.ipynb
│   │   │   ├── python_handler.py
│   │   │   ├── python_model.joblib
│   │   │   └── python_model_test_eval.ipynb
│   │   ├── setup.ipynb
│   │   ├── student_performance_dataset.csv
│   │   ├── test_dataset.csv
│   │   └── train_dataset.csv
│   └── main_model/
│       ├── EDA.ipynb
│       ├── eda_images/
│       │   ├── branch_vs_risk_flag.png
│       │   ├── class_distribution_pie.png
│       │   ├── correlation_heatmap.png
│       │   ├── gender_vs_risk_flag.png
│       │   ├── high_correlation_pairs.png
│       │   ├── high_vif_features.png
│       │   ├── kde_percentile_drop_vs_risk_flag.png
│       │   ├── kde_pred_sem3_vs_risk_flag.png
│       │   ├── kde_sem1_vs_risk_flag.png
│       │   ├── kde_sem2_vs_risk_flag.png
│       │   ├── mutual_information_top25.png
│       │   ├── percentile_drop_vs_risk_flag.png
│       │   ├── pred_sem3_vs_risk_flag.png
│       │   ├── religion_vs_risk_flag.png
│       │   ├── risk_flag_distribution.png
│       │   ├── section1_vs_risk_flag.png
│       │   ├── section2_vs_risk_flag.png
│       │   ├── section3_vs_risk_flag.png
│       │   ├── sem1_vs_risk_flag.png
│       │   ├── sem2_vs_risk_flag.png
│       │   └── top_correlated_features.png
│       ├── model_eda.ipynb
│       ├── model_performance/
│       │   ├── actual_vs_predicted_confusion_matrix.png
│       │   ├── classification_misclass_distribution.png
│       │   ├── grouped_metrics_per_model.png
│       │   ├── shap_comparison_base_models.png
│       │   └── stacking_risk_model_summary.png
│       ├── model_selection.ipynb
│       └── risk_model_metrics.csv
├── LICENSE
├── README.md
├── app.py
├── dataset/
│   ├── Dataset_Documentaion.md
│   ├── branch_department_mapping.csv
│   ├── student_performance_dataset.csv
│   ├── test_dataset.csv
│   └── train_dataset.csv
├── model/
│   ├── de_handler.py
│   ├── de_model.joblib
│   ├── fsd_handler.py
│   ├── fsd_model.joblib
│   ├── main_model_handler.py
│   ├── make_main_model.ipynb
│   ├── math3_handler.py
│   ├── math3_model.joblib
│   ├── python_handler.py
│   ├── python_model.joblib
│   └── stacking_risk_model.joblib
├── pages/
│   ├── Credits.py
│   ├── Data_Insights.py
│   ├── Demo_Predict_Student_Risk.py
│   ├── Feature_Impact.py
│   ├── Feedback.py
│   ├── Predict_Student_Risk.py
│   └── Welcome.py
├── requirements.txt
├── reviews/
│   ├── recent_reviews.json
│   └── word_count.json
└── try.csv

```
---

## Contact

For any questions, collaboration, or feedback:

**Shail K Patel**
LinkedIn: [Shail K Patel](https://www.linkedin.com/in/shailkpatel/)