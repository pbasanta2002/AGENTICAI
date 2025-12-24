# This version removes the complex "Agent" logic (which is used for multi-step reasoning and tool use) 
# and replaces it with a direct "Query Engine" approach. 
# It also uses the global Settings object, which is the modern, cleaner way to configure LlamaIndex.

# This script loads the CSV, converts it to text, creates an in-memory index, and lets you chat with it.

# Key Simplifications Made:
# Removed Agents: Switched from ReActAgent (which "thinks" about which tool to use) to a standard QueryEngine (which just looks up data and answers).

# Global Settings: Used Settings.llm and Settings.embed_model to define models once globally, rather than passing them into every class constructor.

# Dynamic Parsing: The CSV parsing loop now dynamically reads all columns (row.items()) instead of hardcoding specific column names like "Region" or "TotalSale". This makes the code work with any CSV file.

# In-Memory Index: Removed the complex logic for saving/loading the index from disk (StorageContext). This makes the code much shorter and easier to read, though it means it re-indexes every time you run the script (fine for small-to-medium datasets).

import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. Setup & Configuration ---
load_dotenv('c:/codellm/.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure global settings (New LlamaIndex v0.10+ standard)
# This avoids passing 'llm' and 'embed_model' to every function
Settings.llm = Gemini(model_name="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- 2. Load and Prepare Data ---
csv_path = "sales_data.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Please ensure the file exists.")
    exit()

print("Loading data...")
df = pd.read_csv(csv_path)

documents = []
for _, row in df.iterrows():
    # Convert each row into a readable text block for the LLM
    # This dynamic method works for ANY csv columns
    text_content = ", ".join([f"{col}: {val}" for col, val in row.items()])  #[orderid:3245, date:2023-01-01,...]
    documents.append(Document(text=text_content))

# --- 3. Create Index (In-Memory) ---
print(f"Indexing {len(documents)} rows...")
# VectorStoreIndex automatically uses the Settings.llm and Settings.embed_model defined above
index = VectorStoreIndex.from_documents(documents)

# --- 4. Create Query Engine ---
# similarity_top_k=5 means it retrieves the 5 most relevant rows to answer your question
query_engine = index.as_query_engine(similarity_top_k=5)

# --- 5. Simple Query Loop ---
print("\n--- Sales Data RAG Ready ---")
print("Ask questions like 'What was the total sales for Laptops?' or 'exit' to quit.\n")

while True:
    user_input = input("Query: ").strip()
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    if not user_input:
        continue

    try:
        # The query engine retrieves context and generates an answer
        response = query_engine.query(user_input)
        print(f"\nAnswer: {response}\n")
    except Exception as e:
        print(f"Error: {e}")