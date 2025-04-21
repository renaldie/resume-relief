import sys
import importlib.util

if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3

import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import string
import warnings
import tempfile
from dotenv import load_dotenv
import openai
from typing import Dict, List, Any

warnings.filterwarnings("ignore", module=r"chromadb\.types")
warnings.filterwarnings("ignore", module=r"ollama\._types")

## LangChain
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from markitdown import MarkItDown
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings

# Detect if running in Streamlit Cloud
def is_streamlit_cloud():
    return 'STREAMLIT_RUNTIME_GITHUB_TOKEN' in os.environ or 'GITHUB_TOKEN' in os.environ

# Configure models based on environment
if is_streamlit_cloud():
    # Use GitHub Models in Streamlit Cloud
    st.write("Using GitHub Copilot models for analysis")
    try:
        API_HOST = os.getenv("API_HOST", "github")
        client = openai.OpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
        MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
        EMBEDDING_LLM = os.getenv("GITHUB_MODEL", "text-embedding-3-large")
        USE_GITHUB_MODEL = True
    except Exception as e:
        st.error(f"Error initializing GitHub models: {str(e)}")
        st.stop()
else:
    # Use Google API and Nomic locally
    st.write("Using Google API and Nomic for local analysis")
    USE_GITHUB_MODEL = False

st.title("Resume Analysis 🔎")

# Add your resume upload and analysis functionality
st.write("Upload your resume to see AI jobs recommendation.")

# Upload component
uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX)", type=["pdf", "docx"])

# API key handling for both environments
def get_api_key():
    # Try to get API key from Streamlit secrets first
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, AttributeError):
        # Fall back to .env file for local development
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in environment variables or Streamlit secrets")
        return api_key

# Initialize LLM with appropriate API key
def init_llm():
    if USE_GITHUB_MODEL:
        # No need to initialize LLM for GitHub model since we're using the OpenAI client directly
        return None
    else:
        # Use Google API for local development
        api_key = get_api_key()
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-8b",
            temperature=0.8,
            max_tokens=None,
            api_key=api_key,
        )

# Create a function to initialize embeddings and vectorstore
def init_vector_store():
    """Initialize vector store in read-only mode to prevent any modifications to the DB."""
    # Use a location that works in both environments
    is_streamlit = is_streamlit_cloud()
    
    # Choose DB directory based on environment
    if is_streamlit:
        # For Streamlit Cloud deployment
        # We'll need to load the DB from assets or a persistent storage location
        st.error("ChromaDB access not configured for cloud deployment.")
        st.info("Please configure a persistent storage solution for ChromaDB in cloud deployments.")
        return None
    else:
        # Use local directory for development - exactly matching your DB creation path
        db_dir = "./cake_chromadb"
        
        # Check if the DB exists before proceeding
        if not os.path.exists(db_dir) or not os.path.exists(os.path.join(db_dir, "chroma.sqlite3")):
            st.error(f"ChromaDB not found at {db_dir}. Please ensure the database has been created.")
            return None
            
        # Use Nomic embeddings for local development
        embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    try:
        # Initialize vector store in read-only mode
        return Chroma(
            collection_name="cake_db",
            embedding_function=embedding,
            persist_directory=db_dir,
            # Don't create a new collection if it doesn't exist
            create_collection_if_not_exists=False,
            # Read-only mode
            collection_metadata={"read_only": True}
        )
    except Exception as e:
        st.error(f"Error accessing ChromaDB: {str(e)}")
        st.info("Make sure you've already built the database with your job data.")
        return None

def convert_to_md(input_file):
    md = MarkItDown()
    result = md.convert(input_file)
    return result.text_content

def agent_extract_resume(resume, llm):
    resume_template = """"
        You are an expert resume analyzer. Extract the most relevant job search information from the following resume markdown.

        Resume:
        {resume_text}

        Create a concise search query that captures the candidate's:
        1. Primary technical skills (languages, frameworks, tools)
        2. Most recent or significant job roles/titles
        3. Years of experience in key areas
        4. Educational qualifications
        5. Industry expertise
        6. Relevant certifications

        Format your response as a targeted search query that would match this candidate with appropriate job positions.
        Return ONLY the search query text without additional explanations or formatting.
        """

    if USE_GITHUB_MODEL:
        # Use GitHub model directly
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert resume analyzer."},
                    {"role": "user", "content": resume_template.replace("{resume_text}", resume)}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error using GitHub model: {str(e)}")
            return "Error analyzing resume"
    else:
        # Use LangChain with Google API
        # PromptTemplate
        resume_prompt_template = PromptTemplate(
            input_variables=["resume_text"],
            template=resume_template,
        )

        chain = resume_prompt_template | llm | StrOutputParser()
        output = chain.invoke(input={"resume_text": resume})
        return output

def agent_retrieve_jobs(resume, k, category, seniority, vectorstore):
    results = vectorstore.similarity_search_with_relevance_scores(
        query=resume,
        k=k,
        filter={'$and': [
            {'category_major': {'$eq': category}}, 
            {'seniority': {'$eq': seniority}},
        ]}
    )
    
    # Return the results instead of printing
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "score": score * 100,
            "metadata": doc.metadata
        })
    
    return formatted_results

# For Streamlit usage
def streamlit_recommendation(uploaded_file, category, seniority, k=3):
    # Initialize components based on environment
    llm = init_llm()  # Will be None if using GitHub model
    
    with st.spinner("Initializing vector store..."):
        try:
            vectorstore = init_vector_store()
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            return None
    
    # Process the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Convert resume to markdown
        with st.spinner("Converting resume..."):
            resume_md = convert_to_md(temp_file_path)
            if not resume_md:
                st.error("Failed to convert resume to text. Please check the file format.")
                return None
        
        # Extract resume information
        with st.spinner("Analyzing resume..."):
            extracted_resume = agent_extract_resume(resume_md, llm)
            st.write("Extraction complete")
        
        # Get job recommendations
        with st.spinner("Finding matching jobs..."):
            results = agent_retrieve_jobs(extracted_resume, k, category, seniority, vectorstore)
        
        return {
            "extracted_query": extracted_resume,
            "results": results
        }
    except Exception as e:
        st.error(f"Error during recommendation process: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Load available job categories from dataset
@st.cache_data
def job_seniority():
    return ['Assistant', 'Internship', 'Director', 'Entry level',
       'Executive (VP, GM, C-Level)', 'Mid-Senior level']

def job_category():
    return ['Management / Business', 'Sales', 'Marketing / Advertising',
       'Media / Communication', 'Design', 'IT', 'Logistics / Trade', 'HR',
       'Manufacturing', 'Engineering', 'Finance', 'Other',
       'Game Production', 'Customer Service', 'Construction', 'Education',
       'Bio, Medical', 'Catering / Food & Beverage', 'Law',
       'Public Social Work']

# Add Streamlit interface elements
if uploaded_file:
    # Display success message
    st.success("Resume uploaded successfully!")
    
    # Add category and seniority selection
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox(
            "Job Category",
            options=job_category()
        )
    
    with col2:
        seniority = st.selectbox(
            "Experience Level",
            options=job_seniority()
        )
    
    # Add a button to trigger analysis
    if st.button("Analyze Resume", type="primary"):
        # Get job recommendations
        result = streamlit_recommendation(uploaded_file, category, seniority, k=3)
        
        if result:
            # Display the extracted query
            st.subheader("Resume Analysis")
            st.info(f"Based on your resume, we identified the following key elements:\n\n{result['extracted_query']}")
            
            # Display job recommendations
            st.subheader("Job Recommendations")
            if not result['results']:
                st.warning(f"No matching jobs found in the {category} category with {seniority} level. Try a different combination.")
            else:
                for i, job in enumerate(result['results'], 1):

                    job_title = job['metadata'].get('title')
                    name = job['metadata'].get('company_name')
                    job_id = job['metadata'].get('job_id')
                    company_field = job['metadata'].get('company_field')
                    category_major = job['metadata'].get('category_major')
                    employment_type = job['metadata'].get('employment_type')
                    seniority = job['metadata'].get('seniority')
                    location = job['metadata'].get('location')
                    experience = job['metadata'].get('experience')
                    salary_range = job['metadata'].get('salary_range')
                    skills = job['metadata'].get('skills')
                    job_url = job['metadata'].get('job_url')
                    company_url = job['metadata'].get('company_url')
                    
                    with st.expander(f"{job['score']:.1f}% Match | {job_title} in {name}"):
                        # Create tabs for better organization
                        job_tabs = st.tabs(["Details"])                     
                        with job_tabs[0]:
                            st.markdown(f"**Job Title**: {job_title}")
                            st.markdown(f"**Company**: {name}")
                            st.markdown(f"**Company Field**: {company_field}")
                            st.markdown(f"**Job Category**: {category_major}")
                            st.markdown(f"**Employment Type**: {employment_type}")
                            st.markdown(f"**Seniority**: {seniority}")
                            st.markdown(f"**Location**: {location}")
                            st.markdown(f"**Experience**: {experience}")
                            st.markdown(f"**Salary Range**: {salary_range}")
                            st.markdown(f"**Skills**: {skills}")
                            st.markdown(f"**Apply Here**: {job_url}")
                            st.markdown(f"**Company Profile**: {company_url}")
else:
    st.info("Please upload your resume to get job recommendations.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        © 2025 Resume Relief<br>
        Made with ❤️ by <a href="https://renaldi-ega.notion.site" target="_blank">Ren</a>
    </div>
    """, 
    unsafe_allow_html=True
)