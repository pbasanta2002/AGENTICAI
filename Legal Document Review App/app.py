from dotenv import load_dotenv
import os
import streamlit as st
from utils import extract_text_from_pdf, chunk_text, create_vector_store
from chains import build_rag_chain, build_summary_chain

from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="Legal Document Review App", layout="wide")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

uploaded_pdf = st.sidebar.file_uploader("Upload Legal PDF", type=["pdf"])

st.title("⚖️ AI Legal Document Review Application")

# Session states
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""


# === CONFIG ===
load_dotenv('.env')  #load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------------------
# PDF Upload + Processing
# --------------------------------------------
if uploaded_pdf and api_key:
    st.success("📄 PDF Uploaded Successfully!")

    text = extract_text_from_pdf(uploaded_pdf)

    if not text:
        st.error("Unable to extract text. PDF may be scanned or encrypted.")
    else:
        st.session_state.pdf_text = text

        st.subheader("📘 Extracted Text Preview")
        st.text_area("Preview", text[:2000])

        chunks = chunk_text(text)
        st.info(f"Chunked into {len(chunks)} text segments")

        # Vector Store
        vector_store = create_vector_store(chunks, api_key)
        st.session_state.vector_store = vector_store

        st.success("Vector embeddings created successfully!")


# --------------------------------------------
# QUESTION ANSWERING SECTION
# --------------------------------------------
if st.session_state.vector_store:
    st.subheader("🔍 Ask Questions About the Document")

    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        rag_chain = build_rag_chain(st.session_state.vector_store, api_key)
        answer = rag_chain.invoke(question)
        st.write("### Answer:")
        st.write(answer.content)


# --------------------------------------------
# SUMMARIZATION SECTION
# --------------------------------------------
if st.session_state.vector_store:
    st.subheader("📝 Generate Document Summary")

    if st.button("Generate Summary"):
        summary_chain = build_summary_chain(st.session_state.vector_store, api_key)
        summary = summary_chain.invoke("summary")
        st.write("### Summary:")
        st.write(summary.content)
