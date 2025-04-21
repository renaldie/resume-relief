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

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

EMBEDDING = OllamaEmbeddings(
    model="nomic-embed-text:latest")

DB_DIR = "./cake_chromadb"
vectorstore = Chroma(
    collection_name="cake_db",
    embedding_function=EMBEDDING,
    persist_directory=DB_DIR,
    create_collection_if_not_exists=True,
)

jobs_associate = pd.read_csv(f'data/raw/Jobs_associate.csv', encoding="utf-8-sig")
jobs_director = pd.read_csv(f'data/raw/Jobs_director.csv', encoding="utf-8-sig")
jobs_entry_level = pd.read_csv(f'data/raw/Jobs_entry_level.csv', encoding="utf-8-sig")
jobs_executive = pd.read_csv(f'data/raw/Jobs_executive.csv', encoding="utf-8-sig")
jobs_internship_level = pd.read_csv(f'data/raw/Jobs_internship_level.csv', encoding="utf-8-sig")
jobs_mid_senior_level = pd.read_csv(f'data/raw/Jobs_mid_senior_level.csv', encoding="utf-8-sig")

jobs_df = pd.concat([
    jobs_associate, 
    jobs_director, 
    jobs_entry_level, 
    jobs_executive, 
    jobs_internship_level,
    jobs_mid_senior_level
    ], 
    ignore_index=True)

# Rule 1.
# If employment_type is 'Part-time' and title contains intern-related terms and seniority is empty
internship_keywords = ['intern', 'internship', '實習']
mask_part_time = (jobs_df['employment_type'] == 'Part-time') & \
                (jobs_df['seniority'].isna() | (jobs_df['seniority'] == ''))
# Create a boolean mask for titles containing intern-related keywords
mask_intern_title = jobs_df['title'].str.lower().apply(
    lambda x: any(keyword in str(x).lower() for keyword in internship_keywords)
) if 'title' in jobs_df.columns else False
# Apply
jobs_df.loc[mask_part_time & mask_intern_title, 'seniority'] = 'Internship'

# Rule 2.
# If employment_type is 'Internship', always set seniority to 'Internship'
jobs_df.loc[jobs_df['employment_type'] == 'Internship', 'seniority'] = 'Internship'

# Rule 3
# If title contains intern-related terms and seniority is missing, fill with 'Internship'
mask_missing_seniority = jobs_df['seniority'].isna() | (jobs_df['seniority'] == '')
# Check if title contains any internship keywords
mask_intern_title = jobs_df['title'].str.lower().apply(
    lambda x: any(keyword in str(x).lower() for keyword in internship_keywords)
) if 'title' in jobs_df.columns else False
# Apply
jobs_df.loc[mask_missing_seniority & mask_intern_title, 'seniority'] = 'Internship'

# Rule 4
# For remaining missing seniority values, fill with 'Entry level'
still_missing_mask = jobs_df['seniority'].isna() | (jobs_df['seniority'] == '')
jobs_df.loc[still_missing_mask, 'seniority'] = 'Entry level'

# Rule 5
# Create a mask for rows where title contains "Error code"
mask_error_code = jobs_df['title'].str.contains("Error code", case=False, na=False)
jobs_df = jobs_df[~mask_error_code]

documents = []

for index, row in jobs_df.iterrows():
    job_text = f"""
    Job Title: {row['title']}
    Company: {row['company_name']}
    Company Field: {row['company_field']}
    Category Major: {row['category_major']}
    Category Minor: {row['category_minor']}
    Skills: {row['skills']}
    Salary: {row['salary_range']}
    Job Description: {row['job_description']}
    Requirements: {row['requirements']}
    """
    
    doc = Document(
        page_content=job_text,
        metadata={
            "job_id": str(index),
            "title": row['title'],
            "company_name": row['company_name'],
            "company_field": row['company_field'],
            "category_major": row['category_major'],
            "category_minor": row['category_minor'],
            "employment_type": row['employment_type'],
            "seniority": row['seniority'],
            "location": row['location'],
            "experience": row['experience'],
            "salary_range": row['salary_range'],
            "skills": row['skills'],
            "job_url": row['job_url'],
            "company_url": row['company_url'],
        }
    )
    
    documents.append(doc)

vectorstore.add_documents(documents=documents)

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