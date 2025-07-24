import streamlit as st

# Define pages 
home = st.Page("pages/Welcome.py", icon='🏠')

predict_student_risk = st.Page("pages/Predict_Student_Risk.py", icon='🔮') # for prediction
demo = st.Page("pages/Demo_Predict_Student_Risk.py", icon='🎓') # Demo prediction

data_insights = st.Page("pages/Data_Insights.py", icon='📈') # For EDA
model_selection = st.Page("pages/Model_Selection.py", icon='⚙️') # For model selection
model_analysis = st.Page("pages/Model_Analysis.py", icon='🎯') # For SHAP

feedback = st.Page("pages/Feedback.py", icon='💬') # for feedback
credits = st.Page("pages/Credits.py", icon='📜') # for Credits

# Group pages
pg = st.navigation({
    "Home": [home],
    "Predict": [predict_student_risk, demo], # Grouped prediction
    "Insights": [data_insights, model_selection, model_analysis], # Model selection
    "About": [feedback, credits] # Grouped feedback and credits
})

# Run the navigation
pg.run()
