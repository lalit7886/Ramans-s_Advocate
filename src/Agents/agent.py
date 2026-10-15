import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph,START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import tools
load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL","openai/gpt-oss-20b")
Path("data").mkdir(exist_ok=True)
ALLOWED_MODELS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3.8-27b": "qwen/qwen3.8-27b",
    "gpt-oss-safeguard-20b": "openai/gpt-oss-safeguard-20b",
}

SYSTEM_PROMPT = ''''
You are helpful Agentic AI assistant named Advocate Raman.
You can:
1. Answer the normal question.
2. Use tool when needed.
3. Search the uploaded documents using RAG tool.
4. Search the web for current or latest news update related to case.
5. Remember the important user infromation using memory tools.
6. Recall Memory when useful.
7. Remmeber anything important aobut case provided by user.

Rules:
- If user command requires any latest new, recent updates or any thing that requres web search than perform it.
- If user ask about an uploaded document, user search_uploaded_documnts.
- I user asks you to remember something, use remember_this.
-If user ask about previous prefrences or saved facts, use recall_memory.
- When the context given by user and rag system is not enough to give quality answer then user web search.
- Be clear helpful and concise.

'''
def normalize_model_name(model_name: str | None) ->str:
    """
    Validate selected model form frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """
    
    if not model_name:
        return DEFAULT_MODEL
    
    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL
    
    return model_name

def build_agent(model_name: str):
    """
    Build agent with given model name.
    """
    model_name = normalize_model_name(model_name)
    
    llm = ChatGroq(
        model=ALLOWED_MODELS[model_name],
        temperature=0.2,
        max_output_tokens=2048,
        max_input_tokens=4096,
    )
    
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot_node(state:MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        
        return{
            "messages":[response]
        }
        
    tool_node = ToolNode[tools]
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("chatbot",chatbot_node)
    workflow.add_node("tools",tool_node)
    workflow.add_edge(START,"chatbot")
    workflow.add_conditional_edges("chatbot",tools_condition)
    workflow.add_edge("tools","chatbot")
    
    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread = False
    )
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer = checkpointer)

_AGENT_CACHE = {}

def get_agent(model_name: str | None=None):
    """
    Return cached Langgraph agent for selected model.
    If not created yet, create once and reuse it/
    """
    
    selected_model = normalize_model_name(model_name)
    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model]= build_agent(selected_model)
    
    return _AGENT_CACHE[selected_model]
    
    