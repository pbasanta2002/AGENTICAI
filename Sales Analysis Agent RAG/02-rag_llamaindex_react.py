# %pip install llama-index llama-index-llms-groq llama-index-embeddings-huggingface

import os
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import Settings, StorageContext, load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O './uber_2021.pdf' --no-check-certificate
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O './lyft_2021.pdf' --no-check-certificate

uber_pdf_filepath = Path("./uber_2021.pdf")
lyft_pdf_filepath = Path("./lyft_2021.pdf")
index_path_lyft = Path("./storage/lyft")
index_path_uber = Path("./storage/uber")

def setup_environment():
    load_dotenv('c:/codellm/.env')
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
    HF_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")

    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return
    

def get_or_create_index():
    if index_path_lyft.exists():
        print("Loading Lyft index from existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=index_path_lyft)
        lyft_index = load_index_from_storage(storage_context)
    else:
        # load data
        lyft_docs = SimpleDirectoryReader(input_files=[lyft_pdf_filepath]).load_data()
        print("Lyft Docs loaded...building index")
        lyft_index = VectorStoreIndex.from_documents(lyft_docs)
        print("Lyft index created...persisting index")
        lyft_index.storage_context.persist(persist_dir="./storage/lyft")

    if index_path_uber.exists():
        print("Loading Uber index from existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=index_path_uber)
        uber_index = load_index_from_storage(storage_context)
    else:
        uber_docs = SimpleDirectoryReader(input_files=[uber_pdf_filepath]).load_data()
        print("Uber Docs loaded...building index")
        uber_index = VectorStoreIndex.from_documents(uber_docs)
        print("Uber index created...persisting index")
        uber_index.storage_context.persist(persist_dir="./storage/uber")

    print("Lyft and Uber index loaded..")
    return lyft_index, uber_index

def create_agent():
    setup_environment()
    lyft_index, uber_index = get_or_create_index()
    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)
    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=lyft_engine,
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=uber_engine,
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ]
    return ReActAgent.from_tools(query_engine_tools, llm=Settings.llm, verbose=True)

def main():
    agent = create_agent()
    query_history = []
    
    print("Agent ready")
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
    return

if __name__ == "__main__":
    main()