#https://www.youtube.com/watch?v=tcqEUSNCn8I
import warnings
import os
import shutil
import openai
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key Change environment variable name from "OPENAI_API_KEY" to the name given in your .env file.
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
    
    # Load hierarchy JSON files (contains structured hierarchy data)
    hierarchy_documents = load_hierarchy_json()
    documents.extend(hierarchy_documents)
    
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

    # Show sample chunk if available
    if chunks:
        sample_idx = min(10, len(chunks) - 1)
        document = chunks[sample_idx]
        print("Sample chunk:")
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


def load_hierarchy_json():
    """Load and process hierarchy JSON files for RAG search"""
    documents = []
    
    # Path to hierarchy JSON files
    hierarchy_path = "hierarchy"
    
    if not os.path.exists(hierarchy_path):
        print(f"Hierarchy directory not found: {hierarchy_path}")
        return documents
    
    # Look for JSON files
    json_files = [f for f in os.listdir(hierarchy_path) if f.endswith('.json')]
    
    for json_file in json_files:
        json_path = os.path.join(hierarchy_path, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                hierarchy_data = json.load(f)
            
            # Process hierarchy data into searchable documents
            processed_docs = process_hierarchy_json(hierarchy_data, json_path)
            documents.extend(processed_docs)
            
            print(f"Loaded hierarchy JSON: {len(processed_docs)} sections from {json_file}")
            
        except Exception as e:
            print(f"Error loading JSON file {json_file}: {e}")
    
    return documents


def process_hierarchy_json(hierarchy_data, source_file):
    """Process hierarchy JSON data into searchable document chunks"""
    documents = []
    
    nodes = hierarchy_data.get('nodes', {})
    children = hierarchy_data.get('children', {})
    file_type = hierarchy_data.get('file_type', 'unknown')
    
    # Create summary document
    summary_content = "Technical Location Hierarchy Summary\n"
    summary_content += f"Source: {hierarchy_data.get('source_file', 'unknown')}\n"
    summary_content += f"Type: {file_type}\n"
    summary_content += f"Total nodes: {hierarchy_data.get('total_nodes', len(nodes))}\n"
    summary_content += f"Total relationships: {hierarchy_data.get('total_relationships', len(children))}\n\n"
    
    # Add root nodes information
    root_nodes = [node_id for node_id, node_info in nodes.items() if not node_info.get('parent')]
    summary_content += f"Root nodes ({len(root_nodes)}): {', '.join(root_nodes[:10])}\n"
    if len(root_nodes) > 10:
        summary_content += f"... and {len(root_nodes) - 10} more\n"
    
    documents.append(Document(
        page_content=summary_content,
        metadata={
            "source": source_file,
            "type": "hierarchy_summary",
            "file_type": file_type,
            "section": "summary"
        }
    ))
    
    # Create documents for each node with full hierarchy path
    for node_id, node_info in nodes.items():
        # Build full hierarchy path
        hierarchy_path = build_hierarchy_path(node_id, nodes)
        
        content = f"Technical Location: {node_id}\n"
        content += f"Description: {node_info.get('desc', 'No description')}\n"
        content += f"Full Hierarchy Path: {hierarchy_path}\n"
        
        # Add parent information
        parent = node_info.get('parent')
        if parent:
            parent_desc = nodes.get(parent, {}).get('desc', 'No description')
            content += f"Parent: {parent} - {parent_desc}\n"
        
        # Add children information
        node_children = children.get(node_id, [])
        if node_children:
            content += f"Children ({len(node_children)}): "
            child_descriptions = []
            for child in node_children[:5]:  # Show first 5 children
                child_desc = nodes.get(child, {}).get('desc', 'No description')
                child_descriptions.append(f"{child} - {child_desc}")
            content += "; ".join(child_descriptions)
            if len(node_children) > 5:
                content += f"; ... and {len(node_children) - 5} more"
            content += "\n"
        
        documents.append(Document(
            page_content=content,
            metadata={
                "source": source_file,
                "type": "hierarchy_node",
                "file_type": file_type,
                "section": node_id,
                "node_id": node_id,
                "parent": parent,
                "level": len(hierarchy_path.split(" > ")) - 1
            }
        ))
    
    return documents


def build_hierarchy_path(node_id, nodes):
    """Build full hierarchy path from root to node"""
    path = []
    current = node_id
    
    # Prevent infinite loops
    visited = set()
    
    while current and current not in visited:
        visited.add(current)
        node_info = nodes.get(current, {})
        desc = node_info.get('desc', 'No description')
        path.append(f"{current} - {desc}")
        current = node_info.get('parent')
    
    # Reverse to get root-to-leaf order
    path.reverse()
    return " > ".join(path)



if __name__ == "__main__":
    main()
