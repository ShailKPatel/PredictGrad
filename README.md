# PredictGrad

**Overview**

PredictGrad is a machine learning pipeline aimed at forecasting the academic performance of engineering students in their core Semester 3 subjects—Math-3, Digital Electronics (DE), Full Stack Development (FSD), and Python. The goal is to use these predictions to identify students at risk of a significant academic drop and enable early intervention strategies.

**Dataset**

* **Source:** Digitized official academic records from a local engineering college
* **File:** `student_performance_dataset.csv`
* **Size:** 905 students, 56 features (after cleaning)
* **Contents:**

  * Student demographics (Branch, Gender, Religion)
  * Subject-wise theory and practical marks
  * Subject-wise attendance
  * Engineered academic metrics

---

## Problem Statement

The project consists of two main tasks:

1. **Regression Models:**
   Predict raw marks for each Semester 3 core subject (Math-3, DE, FSD, Python) using features from Semesters 1 and 2.

2. **Risk Classification Model:**
   Using the predicted Semester 3 marks, calculate `Sem3_Percentile`, then define a binary flag `Sem3_Risk_Flag = 1` if a student's percentile drops by ≥10 points compared to `Sem2_Percentile`. This flag becomes the target for a classification model.

---

## Data Preprocessing

* **Dropout Filtering:** Removed rows corresponding to dropouts.
* **Anonymization:** Student names and IDs were dropped and replaced with anonymized identifiers.
* **Encoding:** Gender and Religion were inferred from names using GPT, with acknowledgment of possible noise.
* **Cleaning:** All categorical features were one-hot encoded.

---

## Feature Engineering

* **Semester Totals:**

  * `Sem1_Core_Theory_Total` = Math-1 + Physics + Java-1 + SE (Theory only)
  * `Sem2_Core_Theory_Total` = Math-2 + DSJ + DBMS + FEE + Java-2 (Theory only)
  * `Sem3_Core_Theory_Total` = Math-3 + DE + FSD + Python (Theory only)
* **Percentiles:**

  * Calculated percentile scores for each semester total column.

    * `Sem1_Percentile`, `Sem2_Percentile`, `Sem3_Percentile`
* **Output File:**
  All processed columns stored in `student_performance_with_percentiles.csv`.

---

## Regression Modeling

Each Semester 3 subject is modeled independently using a supervised regression approach.

**Evaluation Metric:**
We use **Mean Absolute Error (MAE)** as the primary evaluation metric.

* **Why MAE?**

  * MAE directly represents the average deviation in marks, making it easy to interpret for stakeholders.
  * It treats all errors equally, which is ideal when the goal is to minimize overall prediction drift, not just extreme outliers.
  * Unlike RMSE, MAE avoids exaggerating the impact of a few high-error predictions.

---

## Subject Models (in progress)

Each subject will have a dedicated regression model trained with 5-fold cross-validation. The following subjects are being modeled:

* Math-3 Theory (`Math-3 Theory`)
* Digital Electronics Theory (`DE Theory`)
* Full Stack Development Theory (`FSD Theory`)
* Python Theory (`Python Theory`)

Each model is trained on features from Semesters 1 and 2 including:

* Theory/practical marks
* Attendance percentages
* Engineered totals
* One-hot encoded categorical data

---

## Risk Detection

After predicting Semester 3 subject marks:

1. We calculate `Sem3_Core_Theory_Total` from the predictions.
2. Percentiles are recalculated.
3. A risk flag `Sem3_Risk_Flag` is generated:

   * 1 if `Sem3_Percentile` drops by ≥10 compared to `Sem2_Percentile`
   * 0 otherwise

This flag serves as the label for the final binary classification model aimed at detecting at-risk students.

---

## Contact

For questions, suggestions, or collaboration opportunities, feel free to connect via LinkedIn:
[https://www.linkedin.com/in/shail-k-patel/](https://www.linkedin.com/in/shail-k-patel/)

---
