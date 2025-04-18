import streamlit as st
from pathlib import Path
import nest_asyncio
import asyncio

try:
    nest_asyncio.apply()
except RuntimeError:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


from confluence_rag.src.confluence_rag_llamaindex import ConfluenceRAGWithLlamaIndex
from confluence_rag.config import CHUNKS_DIR, PAGES_DIR

PAGES_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Set paths
data_path = PAGES_DIR / "cleaned_pages.json"
nodes_path = CHUNKS_DIR / 'chunk_nodes.pkl'
index_path = CHUNKS_DIR / 'chunk_index.pkl'

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_rag():
    """Initialize RAG system and load or create index"""

    st.session_state.rag = ConfluenceRAGWithLlamaIndex(
        data_path=data_path,
        chunk_size=512,
        chunk_overlap=128,
    )
        
    if nodes_path.exists() and index_path.exists():
        st.success("Loading pre-computed nodes and index...")
        
        # Load nodes and index
        st.session_state.nodes = st.session_state.rag.load_nodes(nodes_path)
        st.session_state.index = st.session_state.rag.load_index(index_path)
        st.success("Loaded successfully!")

    else:
        st.warning("No existing index found. Creating new nodes and index...")
        
        st.session_state.nodes = st.session_state.rag.process_data()
        
        # Create index
        st.session_state.index = st.session_state.rag.create_index(st.session_state.nodes)
        
        # Save nodes and index
        st.session_state.rag.save_nodes_and_index(
            st.session_state.index, 
            nodes_path, 
            index_path
        )
        st.success("Created and saved successfully!")
    
    st.session_state.initialized = True


def generate_and_display_answer(query):
    """Generate answer using LLM and display it"""
    if not st.session_state.initialized:
        st.error("Please initialize the RAG system first!")
        return
    
    with st.spinner("Generating answer..."):
        answer = st.session_state.rag.generate_answer(query, st.session_state.index) 
        st.subheader("Generated Answer")
        st.write(answer)


st.title("Confluence RAG")

status_placeholder = st.empty()

if not st.session_state.initialized:
    initialize_rag()
else:
    status_placeholder.success("System ready")

st.write("### Ask a question")
query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    generate_and_display_answer(query)

st.write("---")
st.write("LlamaIndex RAG Application")



