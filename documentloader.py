from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_pdf(file_path: str):
    # Load PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents

def create_retriever(documents):
    # Split documents into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    vector_store = FAISS.from_documents(docs, embeddings)

    # Create retriever from vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever
