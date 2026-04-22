"""
router.py

Agentic query router that classifies incoming questions and selectively
activates knowledge graph expansion for relational queries.

Uses few-shot examples drawn from the RAGAS evaluation testset to ground
the classification in the actual domain. Falls back to the static
USE_GRAPH config toggle when routing is disabled.

Evaluated on a 41-question holdout set from the RAGAS testset:
    - DeepSeek-chat: 42.9% three-way, 76.2% binary (SIMPLE vs GRAPH)
    - GPT-5-nano:    33.3% three-way (classifies everything as SIMPLE)
    
DeepSeek selected as production router based on binary accuracy.
"""

import os
import pandas as pd
from pathlib import Path
from config import (
    API_REFORMULATION_MODEL,
    USE_LOCAL_GENERATION,
    LOCAL_GENERATION_MODEL,
    USE_ROUTER,
    USE_GRAPH,
)


def load_example(address=None, n_per_class=3):
    if address is None:
        address = Path(__file__).parent.parent.parent / "outputs" / "ragas_testset.csv"
    path = Path(address)
    df = pd.read_csv(path)
    questions = df[["user_input", "synthesizer_name"]]
    questions = questions.rename(columns={"user_input": "Question", "synthesizer_name": "Type"})
    questions = questions.replace("single_hop_specific_query_synthesizer", "simple")
    questions = questions.replace("multi_hop_abstract_query_synthesizer", "abstract")
    questions = questions.replace("multi_hop_specific_query_synthesizer", "relational")

    sampled = questions.groupby("Type").head(n_per_class).reset_index(drop=True)
    return sampled


# Load examples once at import time
_examples = load_example()


def classify_query(query, examples=_examples):
    """
    Classify a query as SIMPLE, RELATIONAL, or ABSTRACT using few-shot examples
    drawn from the RAGAS evaluation testset.
    """
    from openai import OpenAI

    if USE_LOCAL_GENERATION:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model = LOCAL_GENERATION_MODEL
    else:
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com")
        model = API_REFORMULATION_MODEL

    example_lines = "\n".join(
        f'- "{row.Question}" → {row.Type.upper()}'
        for row in examples.itertuples()
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": f"""Classify the question as SIMPLE, RELATIONAL, or ABSTRACT.

SIMPLE: A question about a single entity with no comparisons, no relationships between things, and no 'how does X affect Y' structure is SIMPLE. If in doubt, classify as SIMPLE.
RELATIONAL: asks about a specific relationship, interaction, or comparison between named entities. Keywords like 'affect', 'relate to', 'compare', 'differ from', 'interact with' signal RELATIONAL.
ABSTRACT: asks about broad themes, trends, or summaries across an entire field or corpus. No specific entities named. Keywords like 'main themes', 'overview', 'general trends' signal ABSTRACT.

Examples from algae research:
{example_lines}

Respond with only one word: SIMPLE or RELATIONAL or ABSTRACT."""
        }, {
            "role": "user",
            "content": query
        }],
        max_completion_tokens=5,  # one word
        temperature=0
    )

    result = response.choices[0].message.content.strip().upper()

    if "ABSTRACT" in result:
        return "ABSTRACT"
    elif "RELATIONAL" in result:
        return "RELATIONAL"
    else:
        return "SIMPLE"


def route_query(query):
    """
    Decide whether to use graph expansion based on query complexity.
    Returns True (use graph) or False (vector only).
    If USE_ROUTER is False in config, falls back to the static USE_GRAPH toggle.
    """
    if not USE_ROUTER:
        return USE_GRAPH

    classification = classify_query(query)

    if classification == "RELATIONAL" or classification == "ABSTRACT":
        return True
    else:
        return False