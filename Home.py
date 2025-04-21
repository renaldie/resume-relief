import sys
import importlib.util

if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import string

# Set page config
st.set_page_config(
    page_title="Resume Relief",
    page_icon="assets/logo.jpg",
    layout="wide",
)

# Main page content
col1, col2 = st.columns([0.2, 1.5])
with col1:
    st.image("assets/logo.jpg", width=100)
    
with col2:
    st.title("Goodbye Jobless 😎")
st.markdown("""
## Welcome to Resume Relief!

This tool helps you understand job requirements across different industries and seniority levels,
and provides insights on how your resume matches those requirements.

**👈 Use the sidebar to navigate between pages:**
- ✨ **Job Insights**: Explore job market requirements
- 🔎 **Resume Analysis**: Upload your resume to get AI recommendations

""")

# Main app
def main():
    # Add a simple credit at the bottom
    st.markdown("---")
    
    cols = st.columns([2, 1, 2])
    with cols[1]:
        st.markdown(
            """
            <div style="text-align: center">
                © 2025 Resume Relief<br>
                Made by <a href="https://renaldi-ega.notion.site" target="_blank">Ren</a> in Hsinchu with ❤️<br>
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()