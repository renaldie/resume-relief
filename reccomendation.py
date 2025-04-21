import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import warnings
import tempfile

warnings.filterwarnings("ignore", module=r"chromadb\.types")
warnings.filterwarnings("ignore", module=r"ollama\._types")

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from markitdown import MarkItDown
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    api_key = get_api_key()
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0.8,
        max_tokens=None,
        api_key=api_key,
    )

# Create a function to initialize embeddings and vectorstore
def init_vector_store():
    # Use a location that works in both environments
    is_streamlit = 'STREAMLIT_SHARING_MODE' in os.environ or hasattr(st, 'session_state')
    
    # Choose DB directory based on environment
    if is_streamlit:
        # Use a temporary directory for Streamlit Cloud
        db_dir = os.path.join(tempfile.gettempdir(), "resume_relief_chromadb")
    else:
        # Use local directory for development
        db_dir = "./cake_chromadb"
    
    # Ensure directory exists
    os.makedirs(db_dir, exist_ok=True)
    
    # Initialize embeddings
    embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    # Initialize vector store
    return Chroma(
        collection_name="cake_db",
        embedding_function=embedding,
        persist_directory=db_dir,
        create_collection_if_not_exists=True,
    )

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
    # Initialize components
    llm = init_llm()
    vectorstore = init_vector_store()
    
    # Process the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Convert resume to markdown
        resume_md = convert_to_md(temp_file_path)
        
        # Extract resume information
        extracted_resume = agent_extract_resume(resume_md, llm)
        
        # Get job recommendations
        results = agent_retrieve_jobs(extracted_resume, k, category, seniority, vectorstore)
        
        return {
            "extracted_query": extracted_resume,
            "results": results
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# For local execution
def local_recommendation(resume_path, category, seniority, k=3):
    # Initialize components
    llm = init_llm()
    vectorstore = init_vector_store()
    
    # Convert resume to markdown
    resume_md = convert_to_md(resume_path)
    
    # Extract resume information
    extracted_resume = agent_extract_resume(resume_md, llm)
    
    # Get job recommendations
    results = agent_retrieve_jobs(extracted_resume, k, category, seniority, vectorstore)
    
    # Print results for local execution
    print(f"Extracted Resume Query: {extracted_resume}")
    print("\nJob Recommendations:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['content']}")
        print(f"   Relevance score: {result['score']:.2f}%")
        print("--" * 30)
    
    return {
        "extracted_query": extracted_resume,
        "results": results
    }

if __name__ == "__main__":
    # This code runs only when executed directly (not through Streamlit)
    resume_path = 'data/example_cv.pdf'
    local_recommendation(resume_path, 'IT', 'Internship', k=2)