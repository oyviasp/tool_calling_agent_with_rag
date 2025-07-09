from langchain.tools import Tool
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import sys
import os

# Load environment variables
load_dotenv()

# Add data processing path for tree_builder import
sys.path.append(os.path.join(os.path.dirname(__file__), 'data', 'data processing'))

#tool that can save  output to a text file with a timestamp
def save_to_txt(data: str, filename: str = "output.txt"):  # Fixed spelling
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
    return success_message  # Don't print here, just return the message

save_tool = Tool(
    name="save_text_to_file",
    function=save_to_txt,
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
        response_text = model.invoke(prompt).content  # Fixed: use invoke instead of predict
        
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"

        return formatted_response
        
    except Exception as e:
        return f"Error searching internal documents: {str(e)}"

# Create the tool properly
RAG_search_in_internal_documents_tool = Tool(
    name="search_internal_documents",
    func=rag_search_function,
    description="Search through internal maintenance documents using RAG (Retrieval Augmented Generation). Use this to find specific information about maintenance procedures, guidelines, or policies."
)

# Technical Location Hierarchy Tool
def technical_location_search(query: str):
    """
    Search for technical locations by ID or description.
    Returns hierarchy information including children and parent locations.
    
    Args:
        query (str): Technical location ID or description to search for
    """
    import pickle
    
    try:
        # Try to load pre-built hierarchy
        hierarchy_data = None
        
        # Load main hierarchy
        main_pickle_path = os.path.join("hierarchy", "hierarchy_main.pkl")
        if os.path.exists(main_pickle_path):
            with open(main_pickle_path, 'rb') as f:
                hierarchy_data = pickle.load(f)
            file_type = "production"
        else:
            return """Error: No pre-built hierarchy database found. 
Please run 'python create_hierarchy_database.py' first to build the hierarchy database from the production Excel file.
This needs to be done once, similar to creating the RAG database."""
        
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
                'match_type': 'exact_id',
                'relevance': 1.0
            })
        
        # Search by description or partial matches
        for node_id, node_info in nodes.items():
            desc = (node_info['desc'] or '').lower()
            
            # Skip if already found exact ID match
            if query in nodes and node_id == query:
                continue
            
            relevance = 0.0
            match_type = 'none'
            
            # Exact description match
            if desc == query_lower:
                relevance = 0.95
                match_type = 'exact_description'
            # Description contains query
            elif query_lower in desc:
                # Higher relevance if query is at the beginning
                if desc.startswith(query_lower):
                    relevance = 0.9
                else:
                    relevance = 0.8
                match_type = 'description_contains'
            # Query contains description (partial match)
            elif desc in query_lower and len(desc) > 2:
                relevance = 0.7
                match_type = 'partial_description'
            # Word-based matching
            else:
                query_words = query_lower.split()
                desc_words = desc.split()
                
                # Count matching words
                matching_words = 0
                for q_word in query_words:
                    if len(q_word) > 2:  # Skip very short words
                        for d_word in desc_words:
                            if q_word in d_word or d_word in q_word:
                                matching_words += 1
                                break
                
                if matching_words > 0:
                    relevance = min(0.6, matching_words / max(len(query_words), len(desc_words)))
                    match_type = 'word_match'
            
            # ID partial matching
            if node_id.lower() == query_lower:
                relevance = max(relevance, 0.9)
                match_type = 'exact_id_case'
            elif query_lower in node_id.lower():
                relevance = max(relevance, 0.6)
                match_type = 'partial_id'
            
            if relevance > 0.3:  # Threshold for inclusion
                matches.append({
                    'id': node_id,
                    'description': node_info['desc'] or '',
                    'parent': node_info['parent'],
                    'match_type': match_type,
                    'relevance': relevance
                })
        
        # Sort by relevance (highest first)
        matches.sort(key=lambda x: x['relevance'], reverse=True)
        
        if not matches:
            # Use LLM-powered intelligent suggestions when no matches found
            llm_suggestions = get_llm_suggestions(query_lower, nodes, children)
            return llm_suggestions
        
        # Format results with explicit line formatting
        result = f"Found {len(matches)} match(es) for '{query}' (using {file_type} hierarchy):\n"
        
        for i, match in enumerate(matches[:3]):  # Limit to 3 matches
            result += f"\n{i+1}. ID: {match['id']}\n"
            result += f"   Description: {match['description']}\n"
            result += f"   Parent: {match['parent'] or 'Root level'}\n"
            
            # Add children information with hierarchy tree
            node_children = children.get(match['id'], [])
            if node_children:
                result += f"\n   Hierarchy Tree (showing {min(len(node_children), 10)} children):\n"
                hierarchy_tree = format_hierarchy_tree(match['id'], nodes, children, max_depth=3, highlight_node=match['id'])
                # Add the tree with consistent indentation
                for line in hierarchy_tree.split('\n'):
                    if line.strip():
                        result += f"   {line}\n"  # Consistent indentation
                
                if len(node_children) > 10:
                    result += f"   ... and {len(node_children) - 10} more children\n"
            else:
                result += "\n   Children: None (leaf node)\n"
            
            # Add hierarchy path with better formatting
            path = []
            current = match['id']
            while current and len(path) < 10:  # Prevent infinite loops
                node_desc = nodes.get(current, {}).get('desc', '')
                path.insert(0, f"{current} ({node_desc})" if node_desc else current)
                current = nodes.get(current, {}).get('parent')
            
            if len(path) > 1:
                result += "\n   Full Hierarchy Path:\n"
                for j, path_item in enumerate(path):
                    indent_str = "    " * (j + 1)  # Clean indentation, 4 spaces per level
                    # Bold the target node in the path too
                    if match['id'] in path_item:
                        # Extract ID and description parts for bold formatting
                        parts = path_item.split(' (', 1)
                        if len(parts) == 2:
                            node_part = parts[0]
                            desc_part = ' (' + parts[1]
                            if node_part == match['id']:
                                path_item = f"\033[1m{node_part}\033[0m{desc_part}"
                    result += f"{indent_str}{path_item}\n"
            
            result += "\n"
        
        if len(matches) > 3:
            result += f"... and {len(matches) - 3} more matches. Please be more specific.\n"
        
        # Add database info
        result += f"\nDatabase info: {hierarchy_data['total_nodes']} total locations, using production data"
        
        return result
        
    except Exception as e:
        return f"Error searching technical locations: {str(e)}"

# Helper function for formatted hierarchy display
def format_hierarchy_tree(node_id, nodes, children, indent=0, max_depth=3, visited=None, highlight_node=None):
    """
    Format a hierarchy tree with clean indentation (no tree characters)
    
    Args:
        node_id: The root node ID to start from
        nodes: Dictionary of all nodes
        children: Dictionary of node children
        indent: Current indentation level
        max_depth: Maximum depth to display
        visited: Set of visited nodes to prevent infinite loops
        highlight_node: Node ID to highlight in bold
    """
    if visited is None:
        visited = set()
    
    if node_id in visited or indent > max_depth:
        return ""
    
    visited.add(node_id)
    
    # Get node info
    node_info = nodes.get(node_id, {})
    desc = node_info.get('desc', '')
    
    # Create clean indentation (4 spaces per level)
    prefix = "    " * indent
    
    # Format current node with bold if it's the highlighted node
    if node_id == highlight_node:
        # Bold formatting using ANSI escape codes
        node_text = f"\033[1m{node_id}\033[0m"  # Bold
        if desc:
            node_text += f" - \033[1m{desc}\033[0m"  # Bold description too
    else:
        node_text = node_id
        if desc:
            node_text += f" - {desc}"
    
    result = f"{prefix}{node_text}\n"
    
    # Add children recursively
    node_children = children.get(node_id, [])
    for child_id in node_children:
        if child_id not in visited and indent < max_depth:
            # Recursively add child with increased indent
            child_result = format_hierarchy_tree(child_id, nodes, children, indent + 1, max_depth, visited.copy(), highlight_node)
            result += child_result
    
    return result

# Create the technical location search tool
technical_location_tool = Tool(
    name="search_technical_locations",
    func=technical_location_search,
    description="Search for technical locations by ID or description. Returns hierarchy information including parent, children, and location path. Use this when users ask about specific technical locations, equipment, or want to navigate the technical hierarchy."
)

#advanced functionality used in technical location search for LLM-powered suggestions thath uses an LLM to intelligently suggest relevant technical locations based on user queries
def get_llm_suggestions(query, nodes, children):
    """
    Use LLM to intelligently understand user intent and suggest relevant technical locations
    """
    try:
        # Create a sample of the hierarchy structure for the LLM
        hierarchy_sample = create_hierarchy_sample(nodes, children)
        
        # Create prompt for LLM
        prompt = f"""
You are an expert at understanding technical location hierarchies in industrial facilities. 
A user searched for "{query}" but no exact matches were found.

Here's a sample of the technical location hierarchy:
{hierarchy_sample}

Based on the user's query "{query}", please:
1. Understand what the user is looking for (equipment type, location, or specific component)
2. Suggest 5-10 relevant technical location IDs from the hierarchy that might match their intent
3. Explain why each suggestion is relevant

Respond in this exact JSON format:
{{
    "analysis": "Brief explanation of what the user seems to be looking for",
    "suggestions": [
        {{
            "id": "technical_location_id",
            "reason": "Why this matches the user's query"
        }}
    ]
}}

Focus on:
- Equipment types (pumps, valves, tanks, cranes, etc.)
- Process areas (electrolysis, smelting, etc.)
- Locations (Høyanger, Sunndal, etc.)
- Similar terminology and synonyms
"""

        # Use LLM to get suggestions
        model = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")  # Use mini for faster/cheaper suggestions
        response = model.invoke(prompt)
        
        # Parse the JSON response
        import json
        try:
            llm_response = json.loads(response.content)
            analysis = llm_response.get('analysis', '')
            suggestions = llm_response.get('suggestions', [])
            
            # Validate that suggested IDs actually exist
            valid_suggestions = []
            for suggestion in suggestions:
                suggested_id = suggestion.get('id', '')
                if suggested_id in nodes:
                    valid_suggestions.append({
                        'id': suggested_id,
                        'description': nodes[suggested_id]['desc'] or '',
                        'reason': suggestion.get('reason', ''),
                        'parent': nodes[suggested_id]['parent']
                    })
            
            # Format the response
            result = f"No exact matches found for '{query}'.\n\n"
            result += f"AI Analysis: {analysis}\n\n"
            
            if valid_suggestions:
                result += "Here are some intelligent suggestions:\n\n"
                for i, suggestion in enumerate(valid_suggestions, 1):
                    result += f"{i}. {suggestion['id']} - {suggestion['description']}\n"
                    result += f"   Why this matches: {suggestion['reason']}\n"
                    if suggestion['parent']:
                        parent_desc = nodes.get(suggestion['parent'], {}).get('desc', '')
                        result += f"   Location: Under {suggestion['parent']} ({parent_desc})\n"
                    result += "\n"
            else:
                result += "The AI couldn't find relevant suggestions in the hierarchy.\n"
                result += "Try using more specific equipment names or location identifiers.\n"
            
            return result
            
        except json.JSONDecodeError:
            # Fallback to traditional suggestions if JSON parsing fails
            return get_fallback_suggestions(query, nodes)
            
    except Exception:
        # Fallback to traditional suggestions if LLM fails
        return get_fallback_suggestions(query, nodes)

def create_hierarchy_sample(nodes, children, max_nodes=100):
    """
    Create a representative sample of the hierarchy for the LLM
    """
    sample_text = "Technical Location Hierarchy Sample:\n\n"
    
    # Find root nodes
    root_nodes = [node_id for node_id, node_info in nodes.items() if not node_info['parent']]
    
    # Show structure for first few root nodes
    nodes_shown = 0
    for root_id in root_nodes[:3]:  # Show max 3 root branches
        if nodes_shown >= max_nodes:
            break
        sample_text += format_hierarchy_sample(root_id, nodes, children, 0, max_depth=4, nodes_shown=nodes_shown, max_nodes=max_nodes)
        nodes_shown += count_nodes_in_subtree(root_id, children, max_depth=4)
    
    # Add some examples of equipment types
    equipment_examples = []
    for node_id, node_info in list(nodes.items())[:200]:  # Sample first 200 nodes
        desc = (node_info['desc'] or '').lower()
        if any(keyword in desc for keyword in ['pump', 'valve', 'tank', 'crane', 'motor', 'sensor']):
            equipment_examples.append(f"{node_id}: {node_info['desc']}")
            if len(equipment_examples) >= 20:
                break
    
    if equipment_examples:
        sample_text += "\nEquipment Examples:\n"
        for example in equipment_examples:
            sample_text += f"- {example}\n"
    
    return sample_text

def format_hierarchy_sample(node_id, nodes, children, indent=0, max_depth=4, nodes_shown=0, max_nodes=100):
    """
    Format a sample of the hierarchy for LLM context
    """
    if indent > max_depth or nodes_shown >= max_nodes:
        return ""
    
    node_info = nodes.get(node_id, {})
    desc = node_info.get('desc', '')
    
    prefix = "  " * indent
    result = f"{prefix}{node_id}: {desc}\n"
    
    # Show only first few children to keep sample manageable
    node_children = children.get(node_id, [])[:3]
    for child_id in node_children:
        if nodes_shown < max_nodes:
            result += format_hierarchy_sample(child_id, nodes, children, indent + 1, max_depth, nodes_shown, max_nodes)
            nodes_shown += 1
    
    if len(children.get(node_id, [])) > 3:
        result += f"{prefix}  ... ({len(children.get(node_id, [])) - 3} more children)\n"
    
    return result

def count_nodes_in_subtree(node_id, children, max_depth=4, current_depth=0):
    """
    Count nodes in a subtree up to a certain depth
    """
    if current_depth >= max_depth:
        return 1
    
    count = 1
    for child_id in children.get(node_id, []):
        count += count_nodes_in_subtree(child_id, children, max_depth, current_depth + 1)
    
    return count

def get_fallback_suggestions(query, nodes):

    """
    Fallback suggestions using traditional fuzzy matching
    """
    suggestions = find_suggestions(query, nodes, max_suggestions=10)
    
    result = f"No matches found for '{query}'."
    if suggestions:
        result += "\n\nDid you mean one of these?\n"
        for i, (node_id, desc, relevance) in enumerate(suggestions, 1):
            result += f"{i}. {node_id}: {desc}\n"
    else:
        result += "\n\nTry using keywords like 'pump', 'valve', 'tank', etc., or provide a technical location ID."
    
    return result

def find_suggestions(query, nodes, max_suggestions=10):
    """
    Find intelligent suggestions based on fuzzy matching and keyword similarity
    """
    suggestions = []
    query_words = query.split()
    
    for node_id, node_info in nodes.items():
        desc = (node_info['desc'] or '').lower()
        
        if not desc:  # Skip nodes without descriptions
            continue
            
        relevance = 0.0
        
        # Fuzzy string matching - check if any query word appears in description
        for q_word in query_words:
            if len(q_word) > 2:  # Skip very short words
                for d_word in desc.split():
                    # Partial word matching
                    if q_word in d_word or d_word in q_word:
                        relevance += 0.3
                    # Edit distance for close matches
                    elif abs(len(q_word) - len(d_word)) <= 2:
                        common_chars = len(set(q_word) & set(d_word))
                        if common_chars >= min(3, len(q_word) - 1):
                            relevance += 0.2
        
        # Bonus for shorter descriptions (more specific)
        if relevance > 0 and len(desc) < 50:
            relevance += 0.1
            
        # Check ID similarity too
        if any(q_word in node_id.lower() for q_word in query_words):
            relevance += 0.2
            
        if relevance > 0.2:  # Minimum threshold
            suggestions.append((node_id, node_info['desc'] or 'No description', relevance))
    
    # Sort by relevance and return top suggestions
    suggestions.sort(key=lambda x: x[2], reverse=True)
    return suggestions[:max_suggestions]