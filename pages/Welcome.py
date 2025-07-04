import streamlit as st

# Title & Tagline
st.title("PredictGrad")
st.markdown("### Academic Risk Prediction for Engineering Students")

# Big Picture Introduction
st.markdown("""
PredictGrad is a machine learning system designed to detect **students at risk of academic decline** — before it happens.

By analyzing student performance data from the first two semesters, the system forecasts outcomes in Semester 3 and flags students likely to suffer a **significant drop in percentile**. These predictions can power timely academic interventions and prevent long-term damage.

This isn't just about marks — it's about **identifying downward trends before they solidify**.
""")

st.markdown("---")

# High-Level Visual Summary
st.header("How It Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Predict Future Marks")
    st.markdown("""
    For each student, PredictGrad estimates raw theory marks in core Semester 3 subjects:
    - Math-3  
    - Digital Electronics (DE)  
    - Full Stack Development (FSD)  
    - Python  
    
    These predictions are grounded in:
    - Past academic scores  
    - Attendance records  
    - Engineered percentile trends  
    """)

with col2:
    st.markdown("#### 2. Detect Risk from Predicted Decline")
    st.markdown("""
    Once predicted marks are in place, PredictGrad:
    - Computes Semester 3 **percentage and percentile**
    - Compares it with the student’s **Semester 2 percentile**
    - Flags students as **'at risk'** if the drop is ≥10 percentile points  

    All of this happens using only data available *before* Semester 3 begins.
    """)

st.markdown("---")

# Why This Matters
st.header("Why This Matters")

st.markdown("""
Academic decline often goes unnoticed until it's too late. Students may:
- Appear to be performing 'fine' based on raw marks
- Hit a cliff in one or two subjects
- Fall behind without formal warning

**PredictGrad looks for subtle signs** — downward trajectories, unexplained underperformance, volatility across subjects — and turns them into early alerts.

This is not a grading system. It's a **risk detection framework.**
""")

st.markdown("---")

# Modeling Snapshot
st.header("Under the Hood (Briefly)")

st.markdown("""
- **Regression Models:** One for each subject (Math-3, DE, FSD, Python)  
  - Algorithm: Voting Regressor (Ridge + Lasso + ElasticNet)  
  - Features: Previous marks, attendance, semester aggregates  
  - Tuning: BayesSearchCV, 5-Fold Repeated CV  

- **Risk Classifier:** Trained to predict **drop ≥10 percentile points**  
  - Stack of CatBoost, BalancedBagging (LGBM), ExtraTrees  
  - Meta model: Logistic Regression  
  - Threshold tuned to balance recall and false alarms  

Result: A system that not only forecasts scores, but recognizes **risk patterns** in academic behavior.
""")

st.markdown("---")

# Dataset Summary (Minimalist)
st.header("The Data")

st.markdown("""
- **905 students** from an engineering college  
- **Semesters 1–3** academic records  
- **56 columns:** marks, attendance, demographics  
- **Sem 1 & 2 attendance** included, **Sem 3 not available**  
- All personal identifiers anonymized

The entire model pipeline is trained on this structured dataset.
""")

st.markdown("---")

# Call to Action (Optional for Deployment Context)
st.header("Ready to Explore?")
st.markdown("""
Use the app to:
- Upload student records
- Run risk analysis
- Understand prediction reasoning with explainable AI
""")
