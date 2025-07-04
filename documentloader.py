from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings


CHROMA_PATH = "chroma"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_pdf("data/file.pdf")  # Replace with your PDF path
    docs = text_splitter(documents)
    save_to_chroma(docs)

def load_pdf(file_path: str):
    # Load PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents

def text_splitter(documents):
    # Split documents into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print(f"Number of chunks created: {len(docs)}")

    document = docs[10]
    print(document.page_content[:500])  # Print first 500 characters of the 10th chunk
    # Create embeddings and vector store
    
    return docs

def save_to_chroma(docs: list[Document]):
    db = Chroma.from_documents(
        docs,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=CHROMA_PATH,
    )
    db.persist()
    print(f"Chroma database created with {len(docs)} documents.")
    return db

if __name__ == "__main__":
    main()