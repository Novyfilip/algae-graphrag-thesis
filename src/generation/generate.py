"""
generation.py

Handles building the context string from reranked chunks,
constructing the generation prompt, and calling OpenAI.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from config import API_GENERATION_MODEL, USE_LOCAL_GENERATION, LOCAL_GENERATION_MODEL

load_dotenv()

def get_client():
    """Initialize and return the appropriate client."""
    if USE_LOCAL_GENERATION:
        return ChatOllama(model=LOCAL_GENERATION_MODEL)
    else:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_context(top_chunks):
    """
    Build a context string from reranked chunks, including metadata headers
    so the generator can cite sources properly.
    
    Args:
        top_chunks: List of (score, document) tuples from reranking
    
    Returns:
        Tuple of (context_string_with_headers, list_of_plain_text_contexts)
    """
    context_parts = []
    contexts_list = []

    for i, (score, doc) in enumerate(top_chunks, 1):
        meta = doc.metadata
        header = (
            f"[{i}] {meta.get('authors', 'Unknown')} "
            f"({meta.get('year', 'n.d.')}). "
            f"{meta.get('title', 'Untitled')}"
        )
        context_parts.append(f"{header}\n{doc.page_content}")
        contexts_list.append(doc.page_content)

    context = "\n\n".join(context_parts)
    return context, contexts_list


def generate_answer(query, context, client, chat_history=None):
    """
    Generate an answer using OpenAI given a query and context string.
    
    Args:
        query: The original user question
        context: The formatted context string with metadata headers
        client: An initialized OpenAI client
    
    Returns:
        The generated answer as a string
    """
    history_str = ""
    if chat_history:
        history_str = "\n".join([
            f"User: {m['content']}" if m['role'] == 'user' 
            else f"Algaebot: {m['content']}" 
            for m in chat_history[-6:]  # last 3 exchanges
        ])
    prompt = f"""
Role: You are an experienced marine biologist and algae cultivation specialist with deep expertise in seagrass ecology, microalgae biotechnology, 
and industrial algae applications.
You prioritize scientific precision, cite 
specific data when available, and clearly distinguish between established 
findings and uncertainties.
Task: Given retrieved passages from scientific literature on algae and 
marine biology, answer the user's question following these steps:

1. First, internally assess which of the provided passages are relevant to the question.
2. Synthesize information from ALL relevant passages into a coherent answer.
   You must draw from multiple sources where possible — do not rely on a single passage.
3. If passages contain conflicting information, acknowledge the conflict and 
   explain both positions.
4. If the provided context is insufficient to fully answer the question, 
   state what is missing.
Domain constraint: Focus on the user's specific industry context.
Output format: Brief paragraph. You MUST cite every passage you use with its 
bracketed number [1], [2], etc. from the context headers. Use multiple citations 
to support your answer. Then at the end of your answer, list all sources of 
retrieved chunks by bracketed number, title, author, year.

The following passages are numbered [1], [2], etc. When citing, use ONLY these 
numbers. Do not invent or extract citations from within the passage text.

Keep your answer grounded strictly in the provided context. Do not introduce 
external knowledge beyond what is given. Match your answer's depth to the 
question's complexity: give concise answers to simple questions, detailed 
analysis to complex ones. Respond in the same language as the user's question.

If the provided context contains NO relevant information to answer the question, 
simply state that the question is outside the scope of the algae research database. 
Do not cite sources that weren't used. Do not list references if none were relevant.

If asked personal questions (your name, preferences, opinions), respond briefly 
as "Algaebot, an algae research assistant" without searching the literature.

IMPORTANT: Respond in the same language as the question below.
Previous conversation:
    {history_str}
Context: {context}
Question: {query}
Answer:"""

    if USE_LOCAL_GENERATION:
        # Ollama path
        response = client.invoke(prompt)
        return response.content
    else:
        # OpenAI path
        response = client.chat.completions.create(
            model=API_GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
