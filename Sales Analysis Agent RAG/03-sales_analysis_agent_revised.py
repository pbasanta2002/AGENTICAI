import os
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import sys

from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding

# --- Configuration ---
INDEX_STORAGE_DIR = "index_storage"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Data Cache
csv_path = "sales_data.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Please ensure the file exists.")
    exit()
_SALES_DF = pd.read_csv("sales_data.csv")

def setup_environment():
    load_dotenv('c:/codellm/.env') # Fallback to default .env
    # GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
    groq_key = os.getenv("GROQ_API_KEY")
    # hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not groq_key:
        raise ValueError("Missing GROQ_API_KEY")
    
    # Configure LlamaIndex Settings globally
    # Settings.llm = Gemini(model_name="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=groq_key)
    # Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # model runs locally
    # Settings.embed_model = HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5",token=hf_token)


def compute_analytics(metric: str, column: str, filter_condition: str = None) -> float:
    """
    Compute metrics like sum and average on sales data with optional filters.
    
    Args:
        metric (str): The metric to calculate ('sum' or 'average').
        column (str): The column name to calculate on (e.g., 'TotalSale').
        filter_condition (str): A pandas query string. The filter could be on Product, Region, Category or Date. (e.g., "Region == 'South'"). 
    """
    try:
        df = _SALES_DF
        
        # Basic cleanup to ensure numeric operations work
        if column in df.columns and df[column].dtype == 'object':
             # Remove currency symbols if present (e.g., '$')
             df[column] = df[column].replace(r'[\$,]', '', regex=True).astype(float)
        
        if filter_condition:
            df = df.query(filter_condition)
            
        if column not in df.columns:
            return 0.0

        if metric == "sum":
            return float(df[column].sum())
        elif metric == "average":
            return float(df[column].mean())
        return 0.0
    except Exception as e:
        logger.error(f"Analytics Error: {e}")
        return 0.0

def get_or_create_index():
    """Load existing index or create a new one from sales data."""
    index_path = Path(INDEX_STORAGE_DIR)
    
    if index_path.exists():
        logger.info("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
        return load_index_from_storage(storage_context)
    
    logger.info("Creating new index...")
    df = _SALES_DF
    
    documents = []
    for _, row in df.iterrows():
        # text = ", ".join([f"{k}: {v}" for k, v in row.items()])
        text = (
            f"OrderID: {row['OrderID']}, Date: {row['Date']}, Region: {row['Region']}, "
            f"Product: {row['Product']}, Category: {row['Category']}, "
            f"Quantity: {row['Quantity']}, UnitPrice: {row['UnitPrice']}, "
            f"TotalSale: {row['TotalSale']}"
        )
        documents.append(Document(text=text))

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    # index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, transformations=[splitter])
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])  #picks up embed_model defined with Settings.embed_model above
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    return index

def create_agent():
    setup_environment()
    
    # 1. Analytics Tool (Statistical)
    analytics_tool = FunctionTool.from_defaults(
        fn=compute_analytics,
        name="analytics_tool",
        description="""Calculates exact sums or averages from sales data. Use this for questions on metrics on sales with optional filtering.
        The sales data has following columns: OrderID, Date, Region, Product, Category, Quantity, UnitPrice, TotalSale
        """
    )

    # 2. RAG Tool (Semantic/Contextual)
    index = get_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=5)
    sales_rag_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sales_context_tool",
        description="Retrieves specific sales records and details. Use this to find trends or look up specific orders."
    )
    return ReActAgent.from_tools([analytics_tool, sales_rag_tool], llm=Settings.llm, verbose=True)

def main():
    agent = create_agent()
    query_history = []
    
    print("\nWelcome to InsightPulse: Your AI-Powered Sales Report Analysis Tool!")
    print("Agent ready with Groq (Llama3) & HF Inference.")
    print("Type 'exit' to quit or 'history' to view log.\n")

    while True:
        user_query = input("Query: ").strip()
        
        if not user_query: continue
        if user_query.lower() == "exit": break
        if user_query.lower() == "history":
            for i, (q, r) in enumerate(query_history[-5:], 1):
                print(f"{i}. {q} -> {r[:50]}...")
            continue

        try:
            print(f"Thinking...")
            response = agent.chat(user_query)
            print(f"Response: {response}\n")
            query_history.append((user_query, str(response)))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

# When you create a VectorStoreIndex in LlamaIndex using the from_documents method, 
# the default node parser/text splitter used automatically is the SentenceSplitter. 

# Key Details of the Default Splitter
# Class Name: SentenceSplitter
# Default Behavior: 
# It attempts to split documents into chunks while respecting sentence and paragraph boundaries, 
# which helps maintain semantic integrity within each chunk.
# Default Parameters:
# chunk_size: The default size for chunks is 1024 tokens.
# chunk_overlap: A default overlap of 20 tokens is used between consecutive chunks.
# separator / paragraphSeparator: It intelligently uses various separators (like newline characters, spaces) to maintain coherent text chunks. 