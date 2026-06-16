import os
import gc
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Global embeddings (loaded once)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_and_index_pdf(pdf_path):
    # Step 1: Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Step 3: Create in-memory vectorstore (no disk, no permission issues)
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        # No persist_directory = stays in memory only
    )

    return vectorstore


def load_existing_index():
    # Since we're in-memory now, nothing to load from disk
    return None


def get_qa_chain(vectorstore):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain