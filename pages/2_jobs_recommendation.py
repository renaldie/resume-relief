# System
import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Streamlit
import streamlit as st
import warnings
import tempfile
warnings.filterwarnings("ignore", module=r"chromadb\.types")
warnings.filterwarnings("ignore", module=r"ollama\._types")

# LangSmith
import langsmith
from langsmith import traceable
# LangChain
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureOpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from markitdown import MarkItDown
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
# LangChain Helper
from typing import Dict, List, Any

# API Keys
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY") or st.secrets.get("AZURE_OPENAI_API_KEY")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT") or st.secrets.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN") or st.secrets.get("ASTRA_DB_APPLICATION_TOKEN")
langsmith_tracing = os.environ.get("LANGSMITH_TRACING") or st.secrets.get("LANGSMITH_TRACING")
langsmith_endpoint = os.environ.get("LANGSMITH_ENDPOINT") or st.secrets.get("LANGSMITH_ENDPOINT")
langsmith_key = os.environ.get("LANGSMITH_API_KEY") or st.secrets.get("LANGSMITH_API_KEY")
langsmith_project = os.environ.get("LANGSMITH_PROJECT") or st.secrets.get("LANGSMITH_PROJECT")

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]


LLM = AzureChatOpenAI(
    azure_endpoint="https://models.inference.ai.azure.com",
    azure_deployment="gpt-4.1-nano",
    openai_api_version="2025-03-01-preview", 
    model_name="gpt-4.1-nano",
    temperature=1,
    api_key=GITHUB_TOKEN,
)

# LLM = ChatOllama(
#     model = "llama3.2:1b",
#     temperature = 0.8,
#     num_predict = 256,
# )

EMBEDDING = AzureOpenAIEmbeddings(
    azure_endpoint="https://resume-relief.openai.azure.com/",
    azure_deployment="text-embedding-3-large",
    openai_api_version="2024-02-01", 
    model="text-embedding-3-large",
    openai_api_key=AZURE_OPENAI_API_KEY,
)

# EMBEDDING = AzureOpenAIEmbeddings(
#     azure_endpoint="https://models.inference.ai.azure.com",
#     azure_deployment="text-embedding-3-large",
#     openai_api_version="2024-02-01", 
#     model="text-embedding-3-large",
#     openai_api_key=GITHUB_TOKEN,
# )

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

@traceable
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

        Format your response as list of keywords in one line that would match this candidate with appropriate job positions.
        Return ONLY the keywords without additional explanations, formatting, or new line.
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

def analyze_resume(uploaded_file):
    """Process the uploaded resume file and extract keywords"""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Convert resume to markdown
        with st.spinner("Converting resume to text..."):
            resume_md = convert_to_md(temp_file_path)
            if not resume_md:
                st.error("Failed to convert resume to text. Please check the file format.")
                return None
        
        # Extract resume information
        # with st.spinner("Getting important keywords...🐝"):
        extracted_keywords = agent_extract_resume(resume_md)
            
        return {
            "resume_md": resume_md,
            "keywords": extracted_keywords
        }
    except Exception as e:
        st.error(f"Error during resume analysis: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def find_job_matches(keywords, category, seniority, k=9):
    """Find job matches based on resume keywords and preferences"""
    try:
        # with st.spinner("Matching up your skills...🧙‍♂️"):
        results = agent_retrieve_jobs(keywords, k, category, seniority, VECTORSTORE)
        return results
    except Exception as e:
        st.error(f"Error retrieving job matches: {str(e)}")
        return []

def refresh_keywords():
    """Callback for the Recreate Keywords button"""
    if uploaded_file:
        result = analyze_resume(uploaded_file)
        if result:
            st.session_state.resume_md = result["resume_md"]
            st.session_state.resume_keywords = result["keywords"]
            st.session_state.resume_analyzed = True

def toggle_edit_mode():
    st.session_state.edit_mode = not st.session_state.edit_mode
    # When entering edit mode, create a copy of current keywords for editing
    if st.session_state.edit_mode:
        st.session_state.edited_keywords = st.session_state.resume_keywords

def save_edited_keywords():
    st.session_state.resume_keywords = st.session_state.edited_keywords
    st.session_state.edit_mode = False

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

# Initialize session state
if "resume_analyzed" not in st.session_state:
    st.session_state.resume_analyzed = False
if "resume_keywords" not in st.session_state:
    st.session_state.resume_keywords = None
if "resume_md" not in st.session_state:
    st.session_state.resume_md = None
if "last_uploaded_file_name" not in st.session_state:
    st.session_state.last_uploaded_file_name = None
if "keywords_generated" not in st.session_state:
    st.session_state.keywords_generated = False
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

st.title("AI Jobs Recommendation in 3 Steps🔎")
st.subheader("1 | Upload Resume ⬆️")
uploaded_file = st.file_uploader(label="Upload your resume (PDF, DOCX, PPTX)", type=["pdf", "docx"], label_visibility='hidden')

if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_file_name:
    st.session_state.resume_analyzed = False
    st.session_state.resume_keywords = None
    st.session_state.resume_md = None
    st.session_state.keywords_generated = False
    st.session_state.last_uploaded_file_name = uploaded_file.name

# STEP 1: Resume Analysis
if uploaded_file and not st.session_state.resume_analyzed:
    if st.button("Analyze Keywords", type="primary", icon='✨', use_container_width=True,):
        result = analyze_resume(uploaded_file)
        if result:
            st.session_state.resume_md = result["resume_md"]
            st.session_state.resume_keywords = result["keywords"]
            st.session_state.resume_analyzed = True
            refresh_keywords()

# STEP 2: Display Keywords
if st.session_state.resume_analyzed:
    st.markdown("---")
    st.subheader("2 | Resume Keywords ✨")
    
    # Display keywords in edit mode or view mode
    if st.session_state.edit_mode:
        # Edit mode - show text area and save/cancel buttons
        st.session_state.edited_keywords = st.text_area(
            "Edit your keywords",
            value=st.session_state.edited_keywords,
            height=250,
            key="keyword_editor"
        )
        
        col1, col2 = st.columns(2)
        if col1.button("Save Changes", type="primary", use_container_width=True):
            save_edited_keywords()
            st.rerun()
        if col2.button("Cancel", type="secondary", use_container_width=True):
            st.session_state.edit_mode = False
            st.rerun()
    else:
        # View mode - show keywords and action buttons
        st.success(f"{st.session_state.resume_keywords}")
        left, middle, right = st.columns(3)
        if left.button("See Job Suggestions", type="primary", use_container_width=True, icon="🧙‍♂️"):
            st.session_state.keywords_generated = True
        middle.button("Refresh Keywords", type="secondary", use_container_width=True, icon="🔄", on_click=refresh_keywords)
        right.button("Manual Edit", type="secondary", icon="📝", use_container_width=True, on_click=toggle_edit_mode)
    
# STEP 3: Display Job Category Selection
if st.session_state.keywords_generated:
    st.markdown("---")
    st.subheader("3 | Voila, We Found These for You! 🧙‍♂️")
    # st.info("Select your dream **Industry** and **Seniority Level**")

    col1, col2 = st.columns(2)
    with col1:
        displayed_options = list(job_category_dict.values())
        actual_options = list(job_category_dict.keys())
        display_index = st.selectbox(
            "Industry",
            options=displayed_options
        )
        selected_index = displayed_options.index(display_index)
        category = actual_options[selected_index]
    
    with col2:
        displayed_options = list(job_seniority_dict.values())
        actual_options = list(job_seniority_dict.keys())
        display_index = st.selectbox(
            "Seniority Level",
            options=displayed_options,
        )
        selected_index = displayed_options.index(display_index)
        seniority = actual_options[selected_index]
    
    # Find matching jobs button
    # if st.button("Find Matching Jobs", type="primary"):
    # Use the existing find_job_matches function
    results = find_job_matches(st.session_state.resume_keywords, category, seniority)
    
    # # Add custom CSS for card styling
    # st.markdown("""
    # <style>
    #     .job-card {
    #         border: 1px solid #ddd;
    #         border-radius: 10px;
    #         padding: 15px;
    #         margin-bottom: 10px;
    #         background-color: white;
    #         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    #         height: 100%;
    #         position: relative;
    #         transition: transform 0.3s;
    #     }
    #     .job-card:hover {
    #         transform: translateY(-5px);
    #         box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    #     }
    #     .match-percentage {
    #         position: absolute;
    #         top: 10px;
    #         right: 10px;
    #         background-color: #4CAF50;
    #         color: white;
    #         border-radius: 20px;
    #         padding: 5px 10px;
    #         font-weight: bold;
    #     }
    #     .job-title {
    #         color: #666;
    #         font-weight: bold;
    #         font-size: 1.1em;
    #         margin-right: 60px;
    #         margin-bottom: 8px;
    #         margin-top: 5px;
    #     }
    #     .company-name {
    #         color: #666;
    #         margin-bottom: 15px;
    #     }
    #     .details-link {
    #         color: #1E88E5;
    #         cursor: pointer;
    #         text-decoration: underline;
    #     }
    # </style>
    # """, unsafe_allow_html=True)

    # # Create rows of 3 cards each
    # for i in range(0, len(results), 3):
    #     cols = st.columns(3)
        
    #     # Create 3 cards per row (or fewer for the last row)
    #     for j in range(3):
    #         if i+j < len(results):
    #             job = results[i+j]
    #             job_title = job['metadata'].get('title')
    #             company_name = job['metadata'].get('company_name')
    #             match_score = job['score']
                
    #             with cols[j]:
    #                 # Use a container for each card
    #                 card = st.container()
    #                 with card:
    #                     # Use HTML for better styling control
    #                     st.markdown(f"""
    #                     <div class="job-card">
    #                         <div class="match-percentage">{match_score:.1f}%</div>
    #                         <div class="job-title">{job_title}</div>
    #                         <div class="company-name">{company_name}</div>
    #                     </div>
    #                     """, unsafe_allow_html=True)
                        
    #                     # Add an expander for details
    #                     with st.expander("Details"):
    #                         st.markdown(f"**Job Title**: {job_title}")
    #                         st.markdown(f"**Company**: {company_name}")
    #                         st.markdown(f"**Company Field**: {job['metadata'].get('company_field')}")
    #                         st.markdown(f"**Job Category**: {job['metadata'].get('category_major')}")
    #                         st.markdown(f"**Employment Type**: {job['metadata'].get('employment_type')}")
    #                         st.markdown(f"**Seniority**: {job['metadata'].get('seniority')}")
    #                         st.markdown(f"**Location**: {job['metadata'].get('location')}")
    #                         st.markdown(f"**Experience**: {job['metadata'].get('experience')}")
    #                         st.markdown(f"**Salary Range**: {job['metadata'].get('salary_range')}")
    #                         st.markdown(f"**Skills**: {job['metadata'].get('skills')}")
                            
    #                         col1, col2 = st.columns(2)
    #                         col1.link_button("Apply Now", job['metadata'].get('job_url'))
    #                         col2.link_button("Company Profile", job['metadata'].get('company_url'))

    # Display job matches
    for i, job in enumerate(results, 1):
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


# # state instructions
# if not uploaded_file:
#     st.info("Upload your resume to begin the magic 💫")
# elif not st.session_state.resume_analyzed:
#     st.info("Click 'Analyze Resume' to extract keywords from your resume.")