from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from datetime import datetime
import os
from tools import search_tool,wiki_tool,save_tool,RAG_search_in_internal_documents_tool

#https://www.youtube.com/watch?v=bTMPwUgLZf0

load_dotenv()

class ResearchResponse(BaseModel):
    """
    A Pydantic model to define the structure of the research response.
    This can be extended with more fields as needed.
    """
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

#print("KEY:", os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(model="gpt-4o")
#llm2= ChatAnthropic(model="claude-3-5-sonnet-20241022")
#response = llm.invoke([HumanMessage(content="What is the meaning of life?")])
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            You are a polite and helpful maintenance assistant thath will help the user with maintenance related questions.
            Answear the user query and use neccesary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
tools = [search_tool, wiki_tool, save_tool, RAG_search_in_internal_documents_tool]  # Add your tools here ....search_tool, wiki_tool
agent = create_tool_calling_agent(
    llm=llm,   
    prompt=prompt,
    tools=tools,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Hi, What can I help you with?")
raw_response = agent_executor.invoke({"query": query})
#print(raw_response)
#print()
structured_response = parser.parse(raw_response.get("output"))
#print(structured_response)
#print()
#print(structured_response.topic)

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e ,"\nRaw response:", raw_response)
#print(response)
#print_pretty_response(response)
