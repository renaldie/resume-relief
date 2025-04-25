import sys
import importlib.util
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import string

def main():
    st.image("assets/logo.jpg", width=400)
    # col1, col2 = st.columns([0.3, 0.5])
    # with col1:
    #     st.image("assets/logo.jpg", use_container_width=True)
        
    # with col2:
    #     st.title("**Resume Relief**")
    # st.header("Analyze Jobs, Resumes, and Skills")
    
    st.markdown("""
    ### AI Job Analysis and Recommendation Platform 
    
    ✨ **Job Analysis** - Discover what employers really want
    
    🔎 **Job Recommendation** - Personal recommendations based on your resume
    """)

if __name__ == "__main__":
    main()