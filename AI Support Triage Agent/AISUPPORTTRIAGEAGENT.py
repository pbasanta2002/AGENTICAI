"""
AI-Powered Customer Support Triage Agent (YNC)
1. High-Level Architecture

User (Streamlit UI)
        |
        v
SupportTriageAgent (Agno)
        |
        +-- File Ingestion (CSV / TXT / PDF)
        |
        +-- Preprocessing & Chunking
        |
        +-- Embedding Generation
        |
        +-- Vector Database (Pinecone)
        |
        +-- LLM Tools
              |-- Intent Classification
              |-- Sentiment & Urgency Detection
              |-- Response Suggestion
              |-- Policy Retrieval
              |-- Supervisor Insights


"""

"""
Technology Stack
Layer	Technology
Agent Framework	Agno (Phidata)
LLM	OpenAI / Gemini
Embeddings	sentence-transformers / Gemini
Vector DB	Pinecone
UI	Streamlit
File Parsing	pandas, pdfplumber
State	Streamlit Session State / JSON
Language	Python 3.10+


"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.tools import Calculator

class SupportTriageAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Customer Support Triage Agent",
            model=OpenAIChat(model="gpt-4o-mini"),
            embedder=SentenceTransformerEmbedder(
                model="all-MiniLM-L6-v2"
            ),
            tools=[
                Calculator()
            ],
            instructions="""
            You are a customer support triage agent.
            - Classify intent (refund, delivery, account, product)
            - Detect sentiment and urgency
            - Suggest professional responses
            - Reference company policies when relevant
            - Highlight escalation cases
            """
        )

import pandas as pd
def load_csv(file):
    df = pd.read_csv(file)
    tickets = []
    for _, row in df.iterrows():
        tickets.append({
            "issue": row["issue"],
            "timestamp": row["created_at"],
            "customer_id": row.get("customer_id", ""),
            "response": row.get("response", "")
        })
    return tickets
import pdfplumber

def load_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def load_txt(file):
    return file.read().decode("utf-8")


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
import pinecone

pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
index = pinecone.Index("support-triage")

def store_embeddings(chunks, embedder):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedder.embed(chunk)
        vectors.append((str(i), embedding, {"text": chunk}))
    index.upsert(vectors)


def semantic_search(query, top_k=5):
    query_vector = embedder.embed(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results

def extract_insights(agent, text):
    prompt = f"""
    Analyze the support ticket:
    {text}

    Return JSON with:
    - intent
    - sentiment
    - urgency (low/medium/high)
    """
    return agent.run(prompt)


def generate_response(agent, issue, policy_context):
    prompt = f"""
    Customer Issue: {issue}
    Relevant Policy: {policy_context}

    Draft a polite, professional support response.
    """
    return agent.run(prompt)



from agno.workflow import Workflow

class SupportTriageWorkflow(Workflow):
    steps = [
        "ingest_logs",
        "preprocess",
        "chunk",
        "embed",
        "extract_insights",
        "generate_response",
        "store_results"
    ]
"""
Upload → Parse → Chunk → Embed → Analyze → Respond → Query

"""

import streamlit as st

st.sidebar.header("Upload Support Files")
csv_files = st.sidebar.file_uploader(
    "Upload CSV Logs", type=["csv"], accept_multiple_files=True
)
pdf_files = st.sidebar.file_uploader(
    "Upload Policy PDFs", type=["pdf"], accept_multiple_files=True
)
query = st.chat_input("Ask about support issues...")
if query:
    results = semantic_search(query)
    response = agent.run(
        f"Answer using these logs: {results}"
    )
    st.chat_message("assistant").write(response)
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append({
    "query": query,
    "response": response
})



"""
Key Business Benefits

✅ Automated ticket categorization
✅ Faster response times
✅ Consistent replies
✅ Supervisor visibility into trends
✅ Reduced SLA violations
✅ Scalable AI-driven support
"""