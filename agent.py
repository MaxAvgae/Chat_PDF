from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
from operator import add as add_messages
from langchain.globals import set_debug
from dotenv import load_dotenv

load_dotenv()
set_debug(True)

def create_agent(retriever):
    @tool
    def retriever_tool(query: str) -> str:
        """Search for relevant documents in the knowledge base based on the query.
        
        Args:
            query: The search query to find relevant documents
            
        Returns:
            A string containing relevant document content or a message if no documents found
        """
        docs = retriever.invoke(query)
        if not docs:
            return "Darling, can't find anything, maybe next time"
        return "\n\n".join([d.page_content[:500] for d in docs])


    tools = [retriever_tool]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    graph = StateGraph(AgentState)

    def call_model(state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        response = llm.invoke(state['messages'])
        return {'messages': [response]}
    
    
    def call_tool(state: AgentState) -> AgentState:
        """Function that check tool usege"""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            result = retriever_tool.invoke(t["args"]["query"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=result)
                )
        return {"messages": results}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    graph.add_node('llm', call_model)
    graph.add_node('tool', call_tool)
   
    graph.set_entry_point('llm')
   
    graph.add_conditional_edges('llm',
        should_continue,
        {True: 'tool', False: END})
    graph.add_edge("tool", "llm")
   
    return graph.compile()