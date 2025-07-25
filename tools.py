from langchain.tools import Tool
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Tool that can save output to a text file with a timestamp
def save_to_txt(data: str, filename: str = "output.txt"):
    """
    Save the output to a text file, including a timestamp.

    Args:
        data (str): The data to save.
        filename (str): The name of the file to save the data to.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_data = (
        f"\n\n--- Output ---\n"
        f"Timestamp: {timestamp}\n"
        f"{'-'*30}\n"
        f"{data}\n"
        f"{'-'*30}\n"
    )
    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_data)
    
    success_message = f"Data saved to {filename}"
    return success_message

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves the research output to a text file with a timestamp.",
)

def rag_search_function(query: str):
    """
    Perform a RAG search in internal documents using ChromaDB.

    Args:
        query (str): The query to search for in the internal documents.
    """
    chroma_path = "chroma" 
    num_docs_to_return = 5

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    try:
        # Prepare the DB.
        db = Chroma(persist_directory=chroma_path, embedding_function=OpenAIEmbeddings())
        
        # Search the DB with the query
        results = db.similarity_search_with_relevance_scores(query, num_docs_to_return)
        if len(results) == 0 or results[0][1] < 0.7:  # threshold
            # No results or low relevance score.
            return "Unable to find matching results in internal documents."
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)

        model = ChatOpenAI()
        response_text = model.invoke(prompt).content
        
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"

        return formatted_response
        
    except Exception as e:
        return f"Error searching internal documents: {str(e)}"

# Create the RAG tool
RAG_search_in_internal_documents_tool = Tool(
    name="search_internal_documents",
    func=rag_search_function,
    description="Search through internal maintenance documents using RAG (Retrieval Augmented Generation). Use this to find specific information about maintenance procedures, guidelines, or policies."
)

# Technical Location Hierarchy Tool - Simplified version
def technical_location_search(query: str):
    """
    Search for technical locations by ID or description.
    Returns hierarchy information including children and parent locations.
    
    Args:
        query (str): Technical location ID or description to search for
    """
    import pickle
    
    try:
        # Load pre-built hierarchy
        main_pickle_path = os.path.join("hierarchy", "hierarchy_main.pkl")
        if not os.path.exists(main_pickle_path):
            return """Error: No pre-built hierarchy database found. 
Please run 'python create_hierarchy_database.py' first to build the hierarchy database from the production Excel file."""
        
        with open(main_pickle_path, 'rb') as f:
            hierarchy_data = pickle.load(f)
        
        nodes = hierarchy_data['nodes']
        children = hierarchy_data['children']
        
        # Search for matches
        matches = []
        query_lower = query.lower().strip()
        
        # Search by exact ID match first
        if query in nodes:
            matches.append({
                'id': query,
                'description': nodes[query]['desc'] or '',
                'parent': nodes[query]['parent'],
                'relevance': 1.0
            })
        
        # Search by description or partial matches
        for node_id, node_info in nodes.items():
            desc = (node_info['desc'] or '').lower()
            
            # Skip if already found exact ID match
            if query in nodes and node_id == query:
                continue
            
            relevance = 0.0
            
            # Exact description match
            if desc == query_lower:
                relevance = 0.95
            # Description contains query
            elif query_lower in desc:
                relevance = 0.8
            # Query contains description (partial match)
            elif desc in query_lower and len(desc) > 2:
                relevance = 0.7
            # ID partial matching
            elif query_lower in node_id.lower():
                relevance = 0.6
            
            if relevance > 0.5:  # Higher threshold for simpler results
                matches.append({
                    'id': node_id,
                    'description': node_info['desc'] or '',
                    'parent': node_info['parent'],
                    'relevance': relevance
                })
        
        # Sort by relevance (highest first)
        matches.sort(key=lambda x: x['relevance'], reverse=True)
        
        if not matches:
            return f"No matches found for '{query}'. Try using a more specific technical location ID or description."
        
        # Format results - simplified output
        result = f"Found {len(matches)} match(es) for '{query}':\n"
        
        for i, match in enumerate(matches[:5]):  # Limit to 5 matches
            result += f"\n{i+1}. ID: {match['id']}\n"
            result += f"   Description: {match['description']}\n"
            result += f"   Parent: {match['parent'] or 'Root level'}\n"
            
            # Show children
            node_children = children.get(match['id'], [])
            if node_children:
                result += f"   Children: {len(node_children)} items\n"
                # Show first few children
                for j, child_id in enumerate(node_children[:5]):
                    child_desc = nodes.get(child_id, {}).get('desc', '')
                    result += f"     - {child_id}: {child_desc}\n"
                if len(node_children) > 5:
                    result += f"     ... and {len(node_children) - 5} more\n"
            else:
                result += "   Children: None\n"
            
            # Show hierarchy path
            path = []
            current = match['id']
            while current and len(path) < 10:
                node_desc = nodes.get(current, {}).get('desc', '')
                path.insert(0, f"{current} ({node_desc})" if node_desc else current)
                current = nodes.get(current, {}).get('parent')
            
            if len(path) > 1:
                result += f"   Hierarchy Path: {' > '.join(path)}\n"
            
            result += "\n"
        
        if len(matches) > 5:
            result += f"... and {len(matches) - 5} more matches. Please be more specific.\n"
        
        return result
        
    except Exception as e:
        return f"Error searching technical locations: {str(e)}"

# Create the technical location search tool
technical_location_tool = Tool(
    name="search_technical_locations",
    func=technical_location_search,
    description="Search for technical locations by ID or description. Returns hierarchy information including parent, children, and location path. Use this when users ask about specific technical locations, equipment, or want to navigate the technical hierarchy."
)
