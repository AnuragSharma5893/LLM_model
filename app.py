import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)
from datetime import datetime

# Custom CSS styling
st.markdown(
    """
    <style>
        /* General styling */
        .main {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stTextInput textarea {
            color: #ffffff !important;
        }
        
        /* Chat bubbles */
        .stChatMessage {
            padding: 12px;
            border-radius: 12px;
            margin: 8px 0;
        }
        .stChatMessage.user {
            background-color: #2d2d2d;
            margin-left: auto;
            max-width: 80%;
        }
        .stChatMessage.ai {
            background-color: #3d3d3d;
            margin-right: auto;
            max-width: 80%;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #2d2d2d;
        }
        
        /* Select box styling */
        .stSelectbox div[data-baseweb="select"] {
            color: white !important;
            background-color: #3d3d3d !important;
        }
        .stSelectbox svg {
            fill: white !important;
        }
        .stSelectbox option {
            background-color: #2d2d2d !important;
            color: white !important;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #4d4d4d;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stButton button:hover {
            background-color: #5d5d5d;
        }
        
        /* Scrollable chat history */
        .chat-history {
            max-height: 400px;
            overflow-y: auto;
            padding: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and caption
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b", "deepseek-r1:32b", "llava:latest", "llama3.2:latest"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown(
        """
        - 🐍 Python Expert
        - 🐞 Debugging Assistant
        - 📝 Code Documentation
        - 💡 Solution Design
        """
    )
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize the chat engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=temperature,
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻", "timestamp": datetime.now().strftime("%H:%M")}
    ]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    st.markdown("### Chat History")
    with st.container():
        for message in st.session_state.message_log:
            with st.chat_message(message["role"]):
                st.markdown(f"**{message['role'].upper()}** ({message['timestamp']})")
                st.markdown(message["content"])

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻", "timestamp": datetime.now().strftime("%H:%M")}
    ]
    st.rerun()

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query, "timestamp": datetime.now().strftime("%H:%M")})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response, "timestamp": datetime.now().strftime("%H:%M")})
    
    # Rerun to update chat display
    st.rerun()