import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

from dotenv import load_dotenv
import pandas as pd
import warnings
warnings.filterwarnings(
    "ignore",
    module=r"chromadb\.types"
)

warnings.filterwarnings(
    "ignore",
    module=r"ollama\._types"
)

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_ollama import OllamaEmbeddings

from langchain_chroma import Chroma

from markitdown import MarkItDown
from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")

LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0.8,
    max_tokens=None,
    api_key=google_api_key,
)

EMBEDDING = OllamaEmbeddings(
    model="nomic-embed-text:latest")

DB_DIR = "./cake_chromadb"
vectorstore = Chroma(
    collection_name="cake_db",
    embedding_function=EMBEDDING,
    persist_directory=DB_DIR,
    create_collection_if_not_exists=True,
)

def convert_to_md(input):
    md = MarkItDown()
    result = md.convert(input)
    return result.text_content

def agent_extract_resume(resume):
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

    chain = resume_prompt_template | LLM | StrOutputParser()
    output = chain.invoke(input={"resume_text": resume})
    return output

def agent_retrieve_jobs(resume, k, category, seniority):
    results = vectorstore.similarity_search_with_relevance_scores(
        query = resume,
        k=k,
        filter={'$and': [
            {'category_major': {'$eq': category}}, 
            {'seniority': {'$eq': seniority}},
            ]}
    )
    for doc, score in results:
        print(f"{doc.page_content} Relevance score is: {score*100:.2f}%")
        print("--"*50)

if __name__ == "__main__":
    resume = convert_to_md('data/example_cv.pdf')
    extracted_resume = agent_extract_resume(resume)
    agent_retrieve_jobs(extracted_resume, 2, 'IT', 'Internship')