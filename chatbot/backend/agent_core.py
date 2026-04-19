import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

AGENT_STORE = dict()
EXACT_MATCH_CACHE = dict()

class ChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self._messages = []

    def add_user_message(self, message:str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message:str):
        self._messages.append(AIMessage(content=message))

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        for message in messages:
            self.add_message(message)

    def clear(self):
        self._messages = []
    
    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        self._messages = value

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in AGENT_STORE:
        AGENT_STORE[session_id] = ChatMessageHistory()
    return AGENT_STORE[session_id]

def get_agent_executor(tools: list = []):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Google API key is not set")
        
    all_tools = []
    all_tools.extend(tools)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=google_api_key)
    
    # Use langgraph for proper tool-calling agent. Instruct it regarding the search tool.
    system_prompt = SystemMessage(
        content="You are a helpful AI assistant. Use the web_search tool ONLY if the user explicitly asks to 'search using web'."
    )
    
    agent_executor = create_react_agent(llm, all_tools, prompt=system_prompt)
    return agent_executor

async def ainvoke_agent(user_id: str, session_id: str, message: str, tools: list = []):
    cache_key = message.strip()
    if cache_key in EXACT_MATCH_CACHE:
        print("Returning cached answer for:", cache_key)
        return EXACT_MATCH_CACHE[cache_key]

    agent_executor = get_agent_executor(tools)
    chat_history_obj = get_session_history(f"{user_id}_{session_id}")
    
    # Build complete message history for standard Langgraph React Agent input
    messages = chat_history_obj.messages + [HumanMessage(content=message)]
    
    response = await agent_executor.ainvoke({"messages": messages})
    
    # Save to history
    chat_history_obj.add_message(HumanMessage(content=message))
    
    final_output = response["messages"][-1].content
    chat_history_obj.add_message(AIMessage(content=final_output))
    
    # Cache the result
    EXACT_MATCH_CACHE[cache_key] = final_output
    
    return final_output
