import streamlit as st

# Define pages 
home = st.Page("pages/Welcome.py", icon='🏠')
predict_student_risk = st.Page("pages/Predict_Student_Risk.py", icon='🔮') # for prediction
demo = st.Page("pages/Demo_Predict_Student_Risk.py", icon='🚄') # for prediction
feature_impact = st.Page("pages/Feature_Impact.py", icon='🎯') # For SHAP
data_insights = st.Page("pages/Data_Insights.py", icon='📈') # For EDA
feedback = st.Page("pages/Feedback.py", icon='💬') # for feedback
credits = st.Page("pages/Credits.py", icon='📜') # for Credits

# Group pages
pg = st.navigation({
    "Home": [home],
    "Analysis & Insights": [predict_student_risk, demo, feature_impact, data_insights], # Grouped prediction, SHAP, and EDA
    "About": [feedback, credits] # Grouped feedback and credits
})

# Run the navigation
pg.run()