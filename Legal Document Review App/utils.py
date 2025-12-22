#Handles: PDF extraction → Chunking → Embeddings → FAISS store
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return None


# ---------------------------------------------
# CHUNKING
# ---------------------------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# ---------------------------------------------
# EMBEDDINGS + VECTOR STORE
# ---------------------------------------------
def create_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store