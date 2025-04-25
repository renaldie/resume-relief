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
    col1, col2 = st.columns([0.2, 2.5])
    with col1:
        st.image("assets/logo.jpg")
        
    with col2:
        st.title("**Resume Relief**")
    # st.header("Analyze Jobs, Resumes, and Skills")
    

    st.markdown("""
    ### AI Job Analysis and Resume Mathing Platform 
    
    ✨ **Job Insights** - Discover what employers really want
    
    🔎 **Resume Analysis** - See how your resume measures up
    """)

if __name__ == "__main__":
    main()