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
        padding-top: 300px;
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
    
    /* Chat container */
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        height: 400px;
        overflow-y: auto;
        background-color: #fafafa;
        margin-bottom: 80px;
    }
    
    .user-message {
        background-color: #e8f5e9;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .bot-message {
        background-color: #ffffff;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
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
    with st.spinner("Loading pipeline... (this may take a moment)"):
        st.session_state.components = setup()

# Chat history display
chat_html = '<div class="chat-container">'
if not st.session_state.messages:
    chat_html += '<p style="color: #888; text-align: center;">Ask me anything about algae research!</p>'
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f'<div class="user-message"><strong>User:</strong> {msg["content"]}</div>'
        else:
            chat_html += f'<div class="bot-message"><strong>Algaebot:</strong> {msg["content"]}</div>'
chat_html += '</div>'

st.markdown(chat_html, unsafe_allow_html=True)

# Input area
# New - chat input with Enter to send, auto-clears
user_input = st.chat_input("Ask about algae research...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Run pipeline
    with st.spinner("Reformulating query, retrieving chunks, and expanding via graph..."):
        answer, contexts, top_chunks, triplets = run_pipeline(user_input, st.session_state.components, st.session_state.messages)
    
    st.session_state.top_chunks = top_chunks
    st.session_state.triplets = triplets  # Store for sidebar
    
    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Rerun to update display
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
        
# Graph visualization
if st.session_state.get("triplets") and len(st.session_state.triplets) > 0:
    from visualization.visualize import create_graph_visualization
    
    st.header("Knowledge Graph Expansion")
    fig = create_graph_visualization(st.session_state.triplets, st.session_state.top_chunks)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('<div class="footer">Made by Filip Nový for University of Southern Denmark, 2026</div>', unsafe_allow_html=True)
