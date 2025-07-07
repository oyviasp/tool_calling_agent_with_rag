from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from ddgs import DDGS

# Load environment variables
load_dotenv()

def web_search_function(query: str):
    """Search the web using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No search results found."
            
            formatted_results = []
            for result in results:
                formatted_results.append(f"Title: {result.get('title', 'N/A')}\nURL: {result.get('href', 'N/A')}\nSnippet: {result.get('body', 'N/A')}")
            
            return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"

search_tool = Tool(
    name="search",
    func=web_search_function,
    description="Search the web for information using DuckDuckGo",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def save_to_txt(data: str, filename: str = "research_output.txt"):  # Fixed spelling
    """
    Save the research output to a text file, including a timestamp.

    Args:
        data (str): The data to save.
        filename (str): The name of the file to save the data to.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_data = (
        f"\n\n--- Research Output ---\n"
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