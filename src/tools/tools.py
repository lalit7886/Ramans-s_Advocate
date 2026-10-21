from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from RAG.rag import retrieve_from_rag
from Database.database import save_memory , search_memory


load_dotenv()
CURRENT_THREAD_ID = "default"

web_search = TavilySearch(
    max_results = 5,
    topic = "general",
    search_depth = "advanced"
)

def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID 
    CURRENT_THREAD_ID = thread_id
    

@tool
def search_uploaded_documents(query:str) ->str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, texts, notes or files.
    """
    
    return retrieve_from_rag(
        query=query,
        thread_id = CURRENT_THREAD_ID
        
    )
    
@tool
def remember_this(memory:str)->str:
    """
    Save an important user prefrence or fact into long-term memory.
    Use this when the user asks you to remeber something.
    Use this when you need to remember anything that you think it will be improtant for future for analysis purpose.
    
    """
    return save_memory(
        
        thread_id = CURRENT_THREAD_ID,
        memory = memory
    )
    
@tool
def recall_memory(query: str) -> str:
    """
    Recall the saved long-term memories about the user or this convesation."""
    
    return search_memory(
        thread_id = CURRENT_THREAD_ID,
        query=query
    )
    
tools = [
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]