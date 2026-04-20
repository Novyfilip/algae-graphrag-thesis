"""
algaebot.py

Streamlit web interface for Algaebot.
Run with: """
#streamlit run algaebot.py --server.port 8502
###

import streamlit as st
from pipeline import setup, run_pipeline

# Page configuration (must be first Streamlit command according to documentation)
st.set_page_config(
    page_title="Demo",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS for seaweed green header and footer
st.markdown("""
<style>
    /* Header */
    .header {
        background-color: #1B2F11;
        padding: 20px;
        text-align: center;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        width: 100%;
    }
    /* Push content below fixed header */
    .block-container {
        padding-top: 70px;
        padding-bottom: 60px;
    }
    .header h1 {
        color: white;
        margin: 0;
        font-size: 2.5em;
    }
    
    /* Footer */
    .footer {
        background-color: #1B2F11;
        padding: 15px;
        text-align: center;
        color: white;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        width: 100%;
    }
    
    /* Chat native container gap fixes */
    .stChatMessage {
        background-color: transparent;
    }
    
    /* User Message Bubble */
    .user-bubble {
        background-color: #2d4a1c;
        color: white;
        padding: 12px 18px;
        border-radius: 12px;
        display: inline-block;
        margin: 5px 0;
    }
    /* Green primary button */
    .stButton > button[kind="primary"] {
        background-color: #1B2F11;
        border-color: #1B2F11;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2d4a1c;
        border-color: #2d4a1c;
    }
    /* Hides Streamlit's default header */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>Demo</h1></div>', unsafe_allow_html=True)

# Initialize session state for chat history and pipeline components
if "messages" not in st.session_state:
    st.session_state.messages = []

if "components" not in st.session_state:
    with st.spinner("Loading pipeline... (this may take a moment on first boot downloading DB)"):
        from download_db import ensure_chromadb_exists
        ensure_chromadb_exists()
        st.session_state.components = setup()
        # Warn the user if the Graph database is turned down!
        if st.session_state.components.get("graph_driver") is None:
            st.toast("Neo4j Database is unreachable! Operating in Vector-Only mode.", icon="⚠️")

# Chat history display using native Streamlit chat bubbles
if not st.session_state.messages:
    st.info("Ask me anything about algae research!")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

# Input area
user_input = st.chat_input("Ask about algae research...")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show spinner while building context
    with st.chat_message("assistant"):
        with st.spinner("Reformulating query, retrieving chunks, and expanding via graph..."):
            answer, contexts, top_chunks, triplets, returned_query = run_pipeline(user_input, st.session_state.components, st.session_state.messages)
        
        st.markdown(answer)
        
        st.session_state.top_chunks = top_chunks
        st.session_state.triplets = triplets
        st.session_state.query = returned_query
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# Sidebar, rendered outside the if block so it persists across reruns
if "top_chunks" in st.session_state and st.session_state.top_chunks:
    with st.sidebar:
        st.header("Retrieved Chunks")
        for score, doc in st.session_state.top_chunks:
            st.markdown(f"**Score: {score:.3f}**")
            st.markdown(f"*{doc.metadata.get('title', 'Untitled')}*")
            st.text(doc.page_content[:200] + "...")
            st.divider()
        
# Graph visualization at the bottom of main view
if st.session_state.get("top_chunks") and st.session_state.get("query"):
    from visualization.visualize import create_graph_visualization
    
    with st.expander("Knowledge Graph Expansion", expanded=True):
        fig = create_graph_visualization(st.session_state.query, st.session_state.top_chunks, st.session_state.triplets)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('<div class="footer">Made by Filip Nový for University of Southern Denmark, 2026</div>', unsafe_allow_html=True)
