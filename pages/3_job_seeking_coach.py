import streamlit as st
from openai import OpenAI
import time
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from retry import retry
import google.generativeai as genai
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import os
load_dotenv()

# API Keys

# Configure API key
#env_path = Path(__file__).parent / ".env"
#load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     #raise EnvironmentError("GOOGLE_API_KEY not set in .env")
#     api_key = st.secrets("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

PROMPT = \
'''
    You are a career-coach assistant. Below are EXAMPLES of the format to follow.
    Each example has a Question followed by the Answer. 
    If a question is outside job-seeking topics, politely refuse. After the examples, answer the new question.

    Example 1:
    Question: How can I improve my resume to stand out?
    Answer: Quantify your achievements, tailor each section to the job description, and include keywords that applicant-tracking systems (ATS) scan for.

    Example 2:
    Question: What's the weather like today?
    Answer: I'm sorry—I can only provide advice on job searching, interview preparation, and skill development. Please ask a question related to those areas.

    Now answer the following:
    Question: {}
    Answer:
'''

SYSTEM_INSTRUCTION = \
'''
You are a career coach specializing in job search strategies, interview preparation, and skill development. 
Answer ONLY questions within that domain; otherwise, politely refuse as in the examples.
'''

# We do not set any safety settings
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Gemini model
def create_gemini_model(system_instruction: str = "You are a helpful assistant"):
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_instruction
    )

# We call the model for inference
@retry(tries=5, delay=1, backoff=2)
def generate_gemini_response(messages: list, system_instruction: str = "You are a helpful assistant") -> str:
    model = create_gemini_model(system_instruction)
    max_output_tokens = 8096
    response = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_output_tokens),
        safety_settings=SAFETY_SETTINGS,
        stream=False
    )
    return response.text

placeholderstr = "Please input your question for the coach"
user_name = "User"
user_image = "https://www.w3schools.com/howto/img_avatar.png"
USER_AVATAR_DEFAULT = "🧑‍💻" 
BOT_AVATAR_DEFAULT = "🤖" 

# def stream_data(stream_str):
#     for word in stream_str.split(" "):
#         yield word + " "
#         time.sleep(0.05)


# st.write("### Ask for advice to our extremely smart coach if you are about to apply for any jobs, and you have questions about the process.")
# st_c_chat = st.container(border=True)

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# else:
#     for msg in st.session_state.messages:
#         if msg["role"] == "user":
#             if user_image:
#                 st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
#             else:
#                 st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
#         elif msg["role"] == "assistant":
#             st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
#         else:
#             try:
#                 image_tmp = msg.get("image")
#                 if image_tmp:
#                     st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
#             except:
#                 st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

# def generate_response(prompt):
#     # output = "Your text in binary is: "
#     # output += ' '.join(format(x, 'b') for x in bytearray(prompt, 'utf-8'))
#     msgs = []
#     msgs.append(PROMPT.format(prompt))
#     response = generate_gemini_response(
#         messages=msgs,
#         system_instruction=SYSTEM_INSTRUCTION
#     )
    
#     return response
    
    
# # Chat function section (timing included inside function)
# def chat(prompt: str):
#     st_c_chat.chat_message("user",avatar=user_image).write(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     response = generate_response(prompt)
#     # response = f"You type: {prompt}"
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st_c_chat.chat_message("assistant").write_stream(stream_data(response))


# if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
#     chat(prompt)


# Streaming data simulation
def stream_data(text, delay=0.01):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)
    yield ""

# Custom CSS 
st.markdown(f"""
<style>
    /* General chat container styling */
    .stChatMessage {{
        margin-bottom: 20px;
    }}

    /* User message styling */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {{
        display: flex;
        flex-direction: row-reverse; 
        text-align: right;
    }}
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stMarkdownContainer"] p {{
        background-color: #0b93f6;
        color: white;
        border-radius: 20px 20px 5px 20px; 
        padding: 10px 15px;
        display: inline-block; 
        max-width: 100%;
        text-align: left; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) .stChatAvatar {{
        margin-left: 8px;
        margin-right: 0;
    }}


    /* Assistant message styling */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div[data-testid="stMarkdownContainer"] p {{
        background-color: #e9ecef; 
        color: #31333F; 
        border-radius: 20px 20px 20px 5px; 
        padding: 10px 15px;
        display: inline-block; 
        max-width: 100%; 
        text-align: left;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) .stChatAvatar {{
        margin-right: 8px;
    }}

</style>
""", unsafe_allow_html=True)


# App Layout
st.title("💬 Job Seeking Coach Chat")
st.write("### Ask for advice to our extremely smart coach if you are about to apply for any jobs, and you have questions about the process.")

# Clear chat
with st.sidebar:
    st.write("Controls")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Chat message container
st_c_chat = st.container(height=500, border=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your career coach. How can I help you today?"} # Initial greeting
    ]

# Display chat messages from history
for msg in st.session_state.messages:
    avatar_to_use = None
    if msg["role"] == "user":
        avatar_to_use = user_image if user_image else USER_AVATAR_DEFAULT
        st_c_chat.chat_message("user", avatar=avatar_to_use).markdown(msg["content"])
    elif msg["role"] == "assistant":
        avatar_to_use = BOT_AVATAR_DEFAULT # Bot avatar
        st_c_chat.chat_message("assistant", avatar=avatar_to_use).markdown(msg["content"])


# Gemini prompting
def generate_response(prompt_text):
    msgs = [PROMPT.format(prompt_text)]
    response = generate_gemini_response(
        messages=msgs,
        system_instruction=SYSTEM_INSTRUCTION
    )
    return response

# Chat Interaction Function
def chat(prompt_text: str):
    # Display user message
    user_avatar_display = user_image if user_image else USER_AVATAR_DEFAULT
    st_c_chat.chat_message("user", avatar=user_avatar_display).markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    # Generate and display assistant response
    with st_c_chat.chat_message("assistant", avatar=BOT_AVATAR_DEFAULT):
        with st.spinner("Coach is thinking..."): # Loading spinner
            response_content = generate_response(prompt_text)
        
        # Typing effect
        st.write_stream(stream_data(response_content))
    
    st.session_state.messages.append({"role": "assistant", "content": response_content})


# Chat Input
placeholderstr = "Ask your question here..."
if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot_input"): # Added a more specific key
    chat(prompt)