# Legal Document Review App (LangChain + LCEL + RAG)

## Features
- Upload legal PDF documents (contracts, NDAs, agreements)
- PDF text extraction (PyPDF2)
- Text chunking + embedding (Gemini text-embedding-004)
- FAISS vector database for semantic search
- RAG-based question answering (Gemini 1.5 Flash)
- Automatic document summarization
- Streamlit web UI with session state
- Secure API key input

## How to Run

1. Clone the repo
2. Install dependencies
   pip install -r requirements.txt

3. Run streamlit app:
   streamlit run app.py

4. Enter your Gemini API key

5. Upload a legal PDF and start querying!

## Sample Questions
- "What are the termination terms?"
- "What is the confidentiality duration?"
- "List the obligations of the vendor."


Uploading PDFs
✔ Extracting text
✔ Chunking + embedding
✔ FAISS vector store
✔ RAG Question Answering
✔ LCEL-based chains
✔ Summarization
✔ Clean Streamlit UI