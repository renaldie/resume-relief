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
        st.title("Land Your Dream Job Faster 😎")
    
    st.markdown("""
    **Resume Relief** analyzes job requirements and matches them to your skills.
    
    ## Get Started:
    
    ✨ **Job Insights** - Discover what employers really want
    
    🔎 **Resume Analysis** - See how your resume measures up
    """)

if __name__ == "__main__":
    main()