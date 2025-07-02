import streamlit as st

# Logo
image = "resources/logo.png"
st.logo(image, size='large')

# ---
# ## 🎖️ Credits
# ---

# Project Contributors Section
with st.container(border=True):
    st.markdown("## Developed By:")

    st.markdown("### **Shail K Patel**")
    st.link_button("Connect on LinkedIn", "https://www.linkedin.com/in/shail-k-patel/", icon="🔗", use_container_width=True)
    st.link_button("Explore on GitHub", "https://github.com/ShailKPatel", icon="🐙", use_container_width=True)
    st.link_button("Visit Portfolio", "https://shailkpatel.github.io/", icon="🌐", use_container_width=True)

# GitHub Repository Section
with st.container(border=True):
    st.markdown("## Project Repository:")
    st.link_button("PredictGrad on GitHub", "https://github.com/ShailKPatel/PredictGrad", icon="💻") # Changed icon to be more relevant
