import sys
import importlib.util
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import string

# Main page content

def main():
    col1, col2 = st.columns([0.2, 1.5])
    with col1:
        st.image("assets/logo.jpg", width=100)
        
    with col2:
        st.title("Goodbye Jobless 😎")
    st.markdown("""
    ## Welcome to Resume Relief!

    This tool helps you understand job requirements across different industries and seniority levels,
    and provides insights on how your resume matches those requirements.

    ### **👈 Use the sidebar to navigate between pages:**

    ✨ **Job Insights**: Explore job market requirements

    🔎 **Resume Analysis**: Upload your resume to get AI recommendations

    """)

if __name__ == "__main__":
    main()