from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from tools import save_tool,RAG_search_in_internal_documents_tool,technical_location_tool

load_dotenv()

# Global variables to track tool usage
used_tools = []
found_sources = []

def track_tool_usage(tool_name, result):
    """Track which tools are used and extract sources"""
    global used_tools, found_sources
    
    if tool_name not in used_tools:
        used_tools.append(tool_name)
    
    # Extract sources from RAG results
    if "search_internal_documents" in tool_name and "Sources:" in str(result):
        try:
            lines = str(result).split('\n')
            for line in lines:
                if 'Sources:' in line:
                    sources_part = line.split('Sources:')[1].strip()
                    # Try to parse as list
                    import ast
                    try:
                        sources_list = ast.literal_eval(sources_part)
                        if isinstance(sources_list, list):
                            for source in sources_list:
                                if source and source not in found_sources:
                                    found_sources.append(str(source))
                    except Exception:
                        # Fallback: treat as string
                        if sources_part and sources_part not in found_sources:
                            found_sources.append(sources_part)
        except Exception:
            pass

# Wrap tools with tracking
def create_tracked_tool(original_tool):
    """Wrap a tool to track its usage"""
    def tracked_func(*args, **kwargs):
        result = original_tool.func(*args, **kwargs)
        track_tool_usage(original_tool.name, result)
        return result
    
    # Create a new tool with the tracked function
    from langchain.tools import Tool
    return Tool(
        name=original_tool.name,
        func=tracked_func,
        description=original_tool.description
    )

# Create tracked versions of tools
tracked_save_tool = create_tracked_tool(save_tool)
tracked_rag_tool = create_tracked_tool(RAG_search_in_internal_documents_tool)
tracked_location_tool = create_tracked_tool(technical_location_tool)

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            You are a polite and helpful maintenance assistant that will help the user with maintenance related questions.
            Answer the user query and use necessary tools. 
            
            When presenting technical location hierarchy information, use clean indentation:
            - Use 4 spaces per hierarchy level
            - No special tree characters
            - Clear visual hierarchy through indentation only
            
            Present your response as natural free text. If you show a hierarchy, draw it out properly with tree structure.
            
            Be helpful and provide complete information based on the tools available to you.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
tools = [tracked_save_tool, tracked_rag_tool, tracked_location_tool]  # Use tracked tools
agent = create_tool_calling_agent(
    llm=llm,   
    prompt=prompt,
    tools=tools,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# Chat functionality with history
def chat_with_agent():
    global used_tools, found_sources
    
    print("🤖 Hi! I'm your maintenance assistant. Type 'quit', 'exit', or 'bye' to end our conversation.")
    print("=" * 60)
    
    chat_history = []
    
    while True:
        try:
            # Reset tracking for each query
            used_tools = []
            found_sources = []
            
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n🤖 Assistant: Goodbye! Feel free to ask me about maintenance anytime.")
                break
            
            if not user_input:
                print("Please enter a question or type 'quit' to exit.")
                continue
            
            print("\n🤖 Assistant: Let me help you with that...")
            print("-" * 40)
            
            # Invoke agent with chat history
            response = agent_executor.invoke({
                "query": user_input,
                "chat_history": chat_history
            })
            
            # Parse and display response
            raw_output = response.get("output", "No output found")
            
            # Extract tool usage information from intermediate steps
            intermediate_steps = response.get("intermediate_steps", [])
            tools_used = []
            rag_sources = []
            
            # Analyze intermediate steps to find used tools and sources
            for step in intermediate_steps:
                if len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    
                    # Get tool name - try different attributes
                    tool_name = None
                    if hasattr(action, 'tool'):
                        tool_name = action.tool
                    elif hasattr(action, 'tool_name'):
                        tool_name = action.tool_name
                    elif hasattr(action, 'name'):
                        tool_name = action.name
                    
                    if tool_name and tool_name not in tools_used:
                        tools_used.append(tool_name)
                    
                    # Extract sources from RAG search results
                    if ("search_internal_documents" in str(tool_name) or 
                        "RAG_search" in str(tool_name)) and "Sources:" in str(observation):
                        
                        # Parse sources from RAG response
                        try:
                            lines = str(observation).split('\n')
                            for line in lines:
                                if 'Sources:' in line:
                                    sources_part = line.split('Sources:')[1].strip()
                                    
                                    # Try to parse as list
                                    import ast
                                    try:
                                        sources_list = ast.literal_eval(sources_part)
                                        if isinstance(sources_list, list):
                                            for source in sources_list:
                                                if source and source not in rag_sources:
                                                    rag_sources.append(str(source))
                                    except Exception:
                                        # Fallback: treat as string
                                        if sources_part and sources_part not in rag_sources:
                                            rag_sources.append(sources_part)
                        except Exception:
                            pass
            
            print("\n🤖 Response:")
            # Display the response with preserved formatting
            response_lines = raw_output.split('\n')
            for line in response_lines:
                if line.strip():
                    # Check if line is indented (hierarchy content) - preserve indentation
                    if line.startswith('    '):
                        print(f"   {line}")  # Add minimal base indentation
                    else:
                        print(f"   {line.strip()}")  # Regular text with basic indentation
                else:
                    print()  # Empty line
            
            # Display tool usage information - use global tracking as fallback
            final_tools_used = tools_used if tools_used else used_tools
            if final_tools_used:
                print(f"\n🔧 Tools used: {', '.join(final_tools_used)}")
            
            # Display RAG sources if any - use global tracking as fallback
            final_rag_sources = rag_sources if rag_sources else found_sources
            if final_rag_sources:
                print("\n📚 Sources from internal documents:")
                for i, source in enumerate(final_rag_sources, 1):
                    # Clean up source path for display
                    clean_source = source.replace('\\', '/').split('/')[-1] if source else source
                    print(f"   {i}. {clean_source}")
            
            # Add to chat history
            chat_history.extend([
                f"Human: {user_input}",
                f"Assistant: {raw_output}"
            ])
            # Keep only last 10 exchanges to prevent context getting too long
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
                
        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ An error occurred: {e}")
            print("Let's try again...")

# Start the chat
if __name__ == "__main__":
    chat_with_agent()
