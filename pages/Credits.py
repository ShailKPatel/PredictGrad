import streamlit as st

st.set_page_config(page_title="Credits", layout="wide")
st.title("🎖️ Credits")

with st.container(border=True):
    "## Made By :"
    "### **Shail K Patel**"
    st.link_button("LinkedIn Profile", "https://www.linkedin.com/in/shail-k-patel/", icon="🔗", use_container_width=True)
    st.link_button("GitHub Profile", "https://github.com/ShailKPatel", icon="🐙", use_container_width=True)
    st.link_button("Portfolio Website", "https://shailkpatel.github.io/", icon="🌐", use_container_width=True)

with st.container(border=True):
    "## GitHub Repository :"
    st.link_button("PredictGrad", "https://github.com/ShailKPatel/PredictGrad", icon="🔗")

with st.container(border=True):
    """
    ## 🛠️ Technologies Used
    - 📌 **Programming & Libraries :** Python, Streamlit, NumPy, Pandas, Plotly, Scikit-Learn, SciPy, PIL, Matplotlib, Seaborn, SHAP, LightGBM, CatBoost, XGBoost, StatsModels, Optuna, Boruta
    - 💻 **IDE & Development :** VS Code, Jupyter Notebook
    - 🌍 **Version Control :** GitHub (Project Repository)
    - 🤖 **Documentation Assistance :** ChatGPT (Generating Documentation)
    - 📦 **Dependency Management :** pip, requirements.txt
    """
