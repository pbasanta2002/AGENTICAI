#RAG - Question Answering Chain (LCEL)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


def build_rag_chain(vector_store, api_key):

    retriever = vector_store.as_retriever(search_type="similarity", k=4)

    # LCEL Prompt
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a legal document assistant. Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer concisely and legally precisely.
"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

    # LCEL Pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
    )

    return chain

#Summarization Chain (LCEL)
def build_summary_chain(vector_store, api_key):

    retriever = vector_store.as_retriever(search_type="similarity", k=6)

    summary_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
You are a senior lawyer. Summarize the following legal document:

Context:
{context}

Provide a 5–7 line summary highlighting:
- Purpose of document
- Key obligations
- Risks / liabilities
- Termination / duration info
- Any important clauses
"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

    chain = (
        {"context": retriever}
        | summary_prompt
        | llm
    )

    return chain
