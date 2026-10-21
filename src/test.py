from Agents.agent import get_agent
from Database.database import init_db
from langchain_core.messages import SystemMessage, HumanMessage
from Database.database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations
)
init_db()
agent = get_agent()

config = {
    "configurable" :{
        "thread_id" : "test_thread_id"
    }
}
for message_chunk , metadata in agent.stream(
    {"messages" : [HumanMessage(content="What is my name?.")]},
    config = config,
    stream_mode = "messages"
):
    if message_chunk.content:
        print(message_chunk.content, end = " ", flush= True)