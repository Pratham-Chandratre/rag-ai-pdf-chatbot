import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


st.set_page_config(page_title="AI PDF Chatbot")

st.title("📄 AI Research Assistant")
st.write("Chat with your research papers using RAG + LangChain")

# 🔑 User enters API Key
groq_api_key = st.text_input(
    "Enter your GROQ API Key",
    type="password",
    placeholder="Paste your GROQ API key here"
)

# Stop execution until API key is provided
if not groq_api_key:
    st.info("Please enter your GROQ API key to start the chatbot.")
    st.stop()


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:

    with st.spinner("Processing PDFs..."):

        documents = []

        pdf_files = [
            "paper.pdf",
            "RAG Paper.pdf"
        ]

        for file in pdf_files:
            if os.path.exists(file):
                loader = PyPDFLoader(file)
                documents.extend(loader.load())
            else:
                st.error(f"{file} not found in the project folder.")
                st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k":4}
        )

        # Initialize Groq LLM using user API key
        llm_model = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm_model,
            retriever,
            return_source_documents=True
        )

        st.session_state.qa_chain = qa_chain

st.success("✅ System Ready!")

# Chat input
query = st.chat_input("Ask a question about the papers...")

if query:

    result = st.session_state.qa_chain({
        "question": query,
        "chat_history": st.session_state.chat_history
    })

    answer = result["answer"]

    st.chat_message("user").write(query)
    st.chat_message("assistant").write(answer)

    st.session_state.chat_history.append((query, answer))

    with st.expander("🔎 Source Documents"):
        for doc in result["source_documents"]:
            st.write(doc.metadata)