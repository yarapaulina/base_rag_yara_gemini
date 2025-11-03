import streamlit as st
import os
from dotenv import load_dotenv
import uuid

# Ensure proper SQLite usage on Linux (for Streamlit Cloud)
if os.name == "posix":
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_utils import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    get_or_create_vector_db,
    DB_DIR
)


load_dotenv()
os.environ["USER_AGENT"] = "myagent"

COLLECTION_NAME = "rag_collection"

def initialize_rag_system():
    if "vector_db" not in st.session_state:
        # Tenta carregar o DB persistido. Se não houver documentos para criar, 
        # docs=None forçará o carregamento se DB_DIR existir.
        try:
            st.session_state.vector_db = get_or_create_vector_db(
                docs=None, 
                collection_name=COLLECTION_NAME
            )
            print("Vector DB persistido carregado na sessão.")
            
        except FileNotFoundError:
            # Não faz nada. Significa que é a primeira execução E não há documentos.
            # O DB será criado quando o usuário fizer o primeiro upload.
            pass

def render_sidebar():
    """Render the sidebar and handle API key & model configuration."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.write("")
        with st.expander("🤖 Model Selection", expanded=True):
            provider = st.radio(
                "Select LLM Provider",
                ["OpenAI", "GROQ", "Anthropic","Gemini"],
                help="Choose which Large Language Model provider to use",
                horizontal=True,
            )
            if provider == "OpenAI":
                model_option = st.selectbox(
                    "Select OpenAI Model",
                    [
                        "gpt-4o-mini",
                        "gpt-4o",
                        "o1",
                        "o1-mini",
                        "o1-preview",
                        "o3-mini",
                        "Custom",
                    ],
                    index=0,
                )
                if model_option == "Custom":
                    model = st.text_input(
                        "Enter your custom OpenAI model:",
                        value="",
                        help="Specify your custom model string",
                    )
                else:
                    model = model_option
            elif provider == "GROQ":
                model = st.selectbox(
                    "Select GROQ Model",
                    [
                        "qwen-2.5-32b",
                        "deepseek-r1-distill-qwen-32b",
                        "deepseek-r1-distill-llama-70b",
                        "llama-3.3-70b-versatile",
                        "llama-3.1-8b-instant",
                        "Custom",
                    ],
                    index=0,
                    help="Choose from GROQ's available models. All these models support tool use and parallel tool use.",
                )
                if model == "Custom":
                    model = st.text_input(
                        "Enter your custom GROQ model:",
                        value="",
                        help="Specify your custom model string",
                    )
            elif provider == "Anthropic":
                model = st.selectbox(
                    "Select Anthropic Model",
                    [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307",
                        "Custom",
                    ],
                    index=0,
                    help="Choose from Anthropic's available models. All these models support tool use and parallel tool use.",
                )
                if model == "Custom":
                    model = st.text_input(
                        "Enter your custom Anthropic model:",
                        value="",
                        help="Specify your custom model string",
                    )
            elif provider == "Gemini": 
                model = st.selectbox(
                     "Select Gemini Model",
                    [
                       "gemini-2.5-flash",
                       "gemini-2.5-pro",
                       "gemini-2.5-flash-multi",
                       "Custom",
                     ],
                     index=0,
                     help="Choose from Google's available Gemini models.",
                 )
                if model == "Custom":
                   model = st.text_input(
                       "Enter your custom Gemini model:",
                       value="",
                       help="Specify your custom model string (e.g., gemini-2.5-flash-vision)",
                    )
            
        with st.expander("🔑 API Keys", expanded=True):
            st.info(
                "API keys are stored temporarily in memory and cleared when you close the browser."
            )
            if provider == "OpenAI":
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API key",
                    help="Enter your OpenAI API key",
                )
                if openai_api_key:
                    os.environ["OPENAI_API_KEY"] = openai_api_key
            elif provider == "GROQ":
                groq_api_key = st.text_input(
                    "GROQ API Key",
                    type="password",
                    placeholder="Enter your GROQ API key",
                    help="Enter your GROQ API key",
                )
                if groq_api_key:
                    os.environ["GROQ_API_KEY"] = groq_api_key
            elif provider == "Anthropic":
                anthropic_api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    placeholder="Enter your Anthropic API key",
                    help="Enter your Anthropic API key",
                )
                if anthropic_api_key:
                    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            elif provider == "Gemini": 
                google_api_key = st.text_input(
                 "Google API Key",
                 type="password",
                 placeholder="Enter your Google API key (for Gemini)",
                 help="Enter your Google API key",
                )
                if google_api_key:
                   os.environ["GOOGLE_API_KEY"] = google_api_key
            

    return {"provider": provider, "model": model}


# --- Page Configuration & Header ---
st.set_page_config(
    page_title="DFD",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.html(
    """<h2 style="text-align: center;"> <i> RAG DFD - Documento de Formalização de Demanda  </i> </h2>"""
)


# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"},
    ]

initialize_rag_system()

# Render sidebar and get selection (provider and model)
selection = render_sidebar()


# Check that API keys are set based on provider
if selection["provider"] == "OpenAI":
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started")
        st.stop()
elif selection["provider"] == "GROQ":
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("⚠️ Please enter your GROQ API key in the sidebar to get started")
        st.stop()
elif selection["provider"] == "Anthropic":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning(
            "⚠️ Please enter your Anthropic API key in the sidebar to get started")
        st.stop()
elif selection["provider"] == "Gemini": 
    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("⚠️ Please enter your Google API key in the sidebar to get started")
        st.stop()

# --- Sidebar Additional Controls ---
with st.sidebar:
    cols0 = st.columns(2)
    with cols0[0]:
        # Enable "Use RAG" only if a vector DB exists (i.e. if a document has been loaded)
        is_vector_db_loaded = (
            "vector_db" in st.session_state and st.session_state.vector_db is not None
        )
        st.toggle(
            "Use RAG",
            value=is_vector_db_loaded,
            key="use_rag",
            disabled=not is_vector_db_loaded,
        )
    with cols0[1]:
        st.button(
            "Clear Chat",
            on_click=lambda: st.session_state.messages.clear(),
            type="primary",
        )

    st.header("RAG Sources:")
    st.file_uploader(
        "📄 Upload a document",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )
    st.text_input(
        "🌐 Introduce a URL",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )
    with st.expander(
        f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"
    ):
        st.write([] if not is_vector_db_loaded else st.session_state.rag_sources)

    # Initialize the appropriate LLM stream based on the provider
    if selection["provider"] == "OpenAI":
        llm_stream = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name=selection["model"],
            temperature=0,
            streaming=True,
        )
    elif selection["provider"] == "GROQ":
        llm_stream = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model=selection["model"],
            temperature=0,
            streaming=True,
        )
    elif selection["provider"] == "Anthropic":
        llm_stream = ChatAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=selection["model"],
            temperature=0,
            streaming=True,
        )
    elif selection["provider"] == "Gemini": 
        llm_stream = ChatGoogleGenerativeAI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            model=selection["model"],
            temperature=0,
            streaming=True,
        )


# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Chat Input and Streaming Response ---
if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Convert the chat history to LangChain message objects
        messages = [
            (
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
            )
            for m in st.session_state.messages
        ]

        # Choose the appropriate stream, CAPTURING the full response
        if not st.session_state.use_rag:
            full_response = st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            full_response = st.write_stream(stream_llm_rag_response(llm_stream, messages))

    # Adiciona a resposta completa ao histórico APÓS o streaming.
    st.session_state.messages.append({"role": "assistant", "content": full_response})