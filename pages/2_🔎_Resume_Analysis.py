import sys
import importlib.util
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
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from markitdown import MarkItDown
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings

load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY") or st.secrets.get("AZURE_OPENAI_API_KEY")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT") or st.secrets.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN") or st.secrets.get("ASTRA_DB_APPLICATION_TOKEN")

st.title("Resume Analysis 🔎")

# Add your resume upload and analysis functionality
# st.write("Upload your resume to see AI jobs recommendation.")

# Upload component
uploaded_file = st.file_uploader(label="Upload your resume (PDF, DOCX)", type=["pdf", "docx"])

LLM = AzureChatOpenAI(
    azure_endpoint="https://models.inference.ai.azure.com",
    azure_deployment="gpt-4.1-nano",
    openai_api_version="2025-03-01-preview", 
    model_name="gpt-4.1-nano",
    temperature=1,
    api_key=GITHUB_TOKEN,
)

EMBEDDING = AzureOpenAIEmbeddings(
    azure_endpoint="https://resume-relief.openai.azure.com/",
    azure_deployment="text-embedding-3-large",
    openai_api_version="2024-02-01", 
    model="text-embedding-3-large",
    openai_api_key=AZURE_OPENAI_API_KEY,
)

VECTORSTORE = AstraDBVectorStore(
    collection_name="cake_db",
    embedding=EMBEDDING,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace="cake_db",
)

def convert_to_md(input_file):
    md = MarkItDown()
    result = md.convert(input_file)
    return result.text_content

def agent_extract_resume(resume):
    resume_template = """"
        You are an expert resume analyzer. Extract the most relevant job search keywords from the following resume markdown.

        Resume:
        {resume_text}

        Create a concise keywords that captures the candidate's:
        1. Primary technical skills (languages, frameworks, tools)
        2. Most recent or significant job roles/titles
        3. Years of experience in key areas
        4. Educational qualifications
        5. Industry expertise
        6. Relevant certifications

        Format your response as list of keywords that would match this candidate with appropriate job positions.
        Return ONLY the keywords without additional explanations or formatting.
        """

    resume_prompt_template = PromptTemplate(
        input_variables=["resume_text"],
        template=resume_template,
    )

    chain = resume_prompt_template | LLM | StrOutputParser()
    output = chain.invoke(input={"resume_text": resume})
    return output

def agent_retrieve_jobs(resume, k, category, seniority, VECTORSTORE):
    results = VECTORSTORE.similarity_search_with_relevance_scores(
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
def streamlit_recommendation(uploaded_file, category, seniority, k=10):
    # Process the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
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
            extracted_resume = agent_extract_resume(resume_md)
            
        # Get job recommendations if possible
        results = []
        with st.spinner("Finding matching jobs..."):
            results = agent_retrieve_jobs(extracted_resume, k, category, seniority, VECTORSTORE)
        
        return {
            "extracted_query": extracted_resume,
            "results": results,
            "vectorstore_available": VECTORSTORE is not None
        }
    except Exception as e:
        st.error(f"Error during recommendation process: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# # Load available job categories from dataset
# @st.cache_data
job_seniority_dict = {
    'Internship': '🎓 Internship',
    'Entry level': '🌱 Entry Level',
    'Assistant': '🔍 Assistant',
    'Mid-Senior level': '⚙️ Mid-Senior Level',
    'Director': '🚀 Director',
    'Executive (VP, GM, C-Level)': '👑 Executive (VP, GM, C-Level)',
    }

job_category_dict = {
    "Bio, Medical": "🧬 Bio / Medical",
    "Catering / Food & Beverage": "🍔 Catering / Food & Beverage",
    "Construction": "🏗️ Construction",
    "Customer Service": "🛎️ Customer Service",
    "Design": "🎨 Design",
    "Education": "📚 Education",
    "Engineering": "🔧 Engineering",
    "Finance": "💵 Finance",
    "Game Production": "🎮 Game Production",
    "HR": "👩‍💼 HR",
    "IT": "💻 IT",
    "Law": "⚖️ Law",
    "Logistics / Trade": "🚚 Logistics / Trade",
    "Management / Business": "👔 Management / Business",
    "Manufacturing": "🏭 Manufacturing",
    "Marketing / Advertising": "📢 Marketing / Advertising",
    "Media / Communication": "📡 Media / Communication",
    "Public Social Work": "💞 Public Social Work",
    "Sales": "📈 Sales",
    "Other": "💅 Other",
    }

# Add Streamlit interface elements
if uploaded_file:
    # Display success message
    st.success("Resume uploaded successfully! Select job category and seniority level to get recommendations.")
    
    # Add category and seniority selection
    col1, col2 = st.columns(2)
    
    with col1:
        displayed_options = list(job_category_dict.values())
        actual_options = list(job_category_dict.keys())
        
        # Get the index of the selected display value, then use that to get the corresponding key
        display_index = st.selectbox(
            "Job Category",
            options=displayed_options,
            format_func=lambda x: x  # This isn't necessary but makes the intent clearer
        )
        
        # Find the index of the selected display option and use it to get the corresponding key
        selected_index = displayed_options.index(display_index)
        category = actual_options[selected_index]
    
    with col2:
        displayed_options = list(job_seniority_dict.values())
        actual_options = list(job_seniority_dict.keys())
        
        # Get the index of the selected display value, then use that to get the corresponding key
        display_index = st.selectbox(
            "Experience Level",
            options=displayed_options,
            format_func=lambda x: x  # This isn't necessary but makes the intent clearer
        )
        
        # Find the index of the selected display option and use it to get the corresponding key
        selected_index = displayed_options.index(display_index)
        seniority = actual_options[selected_index]
    
    # Add a button to trigger analysis
    if st.button("Analyze Resume", type="primary"):
        # Get job recommendations
        result = streamlit_recommendation(uploaded_file, category, seniority, k=10)
        
        if result:
            # Display the extracted query

            st.markdown("---")
            st.subheader("Resume Keywords")
            st.info(f"Based on your resume, we identified the following key elements:\n\n{result['extracted_query']}")
            
            # Display job recommendations if available
            if result.get("vectorstore_available", False):
                st.markdown("---")
                st.subheader(f"Job Recommendations for {seniority} in {category}")
                if not result['results']:
                    st.warning(f"No matching jobs found in the {category} category with {seniority} level. Try a different combination.")
                else:
                    for i, job in enumerate(result['results'], 1):
                        job_title = job['metadata'].get('title')
                        name = job['metadata'].get('company_name')
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
                st.info("The resume analysis is complete, but job matching requires a local database setup.")   

else:
    st.info("Please upload your resume to get job recommendations.")