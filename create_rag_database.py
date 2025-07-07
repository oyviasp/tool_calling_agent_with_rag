#https://www.youtube.com/watch?v=tcqEUSNCn8I
import warnings
import os
import shutil
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/pdf"  # Will search in both books and pdf subdirectories


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    documents = []
    
    # Load PDF files from pdf directory
    pdf_path = os.path.join(DATA_PATH, "pdf")
    if os.path.exists(pdf_path):
        pdf_loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)
        print(f"Loaded {len(pdf_documents)} PDF files from {pdf_path}")
    
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    documents.extend(pdf_documents)
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
