"""
Example 3:

This module demonstrates how to build a "smart" AI agent using LangChain that can 
access external tools, such as the internet, to retrieve up-to-date information. 
It uses the Tavily search API to allow the agent to look up current facts, 
weather, or any other information available online, making it much more powerful 
than a standard LLM that only knows information up to its training cutoff.
"""

import os
import sys  
from load_env import configure_environment

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage,BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_tavily import TavilySearch
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.box import ASCII

# Set environment variable to force UTF-8 before Console is fully bound
os.environ["PYTHONIOENCODING"] = "utf-8"
console = Console()


AGENT_STORE = dict()

def create_session_config(user_id:str, session_id: str):

    return {"configurable": {"session_id": f"{user_id}_{session_id}"}}

def get_llm(apikey, model="gpt-4o", temperature=0,tools=None):
    """
    Initializes and returns a ChatOpenAI language model instance.

    Args:
        apikey (str): The OpenAI API key used for authentication.
        model (str, optional): The model version to use. Defaults to "gpt-4o".
        temperature (float, optional): Sampling temperature. Defaults to 0 for deterministic outputs.

    Returns:
        ChatOpenAI: A configured LangChain ChatOpenAI LLM object.
    """
    if tools:
        llm = ChatOpenAI(model=model, temperature=temperature, api_key=apikey)
        return llm.bind_tools(tools)
    return ChatOpenAI(model=model, temperature=temperature, api_key=apikey)

class ChatMessageHistory(BaseChatMessageHistory):
    """
    A simple in-memory implementation of a chat message history for LangChain.
    
    This class stores the conversation history as a list of BaseMessage objects. 
    It tracks the sequence of interactions between the user and the AI, which 
    allows the agent to recall context during ongoing conversations.
    """
    def __init__(self):
        """Initializes an empty chat message history."""
        self._messages = []

    def add_user_message(self, message:str):
        """
        Appends a user input string to the message history.
        
        Args:
            message (str): The user's message text.
        """
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message:str):
        """
        Appends an AI-generated string to the message history.
        
        Args:
            message (str): The AI's response text.
        """
        self._messages.append(AIMessage(content=message))

    def add_message(self, message: BaseMessage) -> None:
        """
        Appends a single LangChain BaseMessage (e.g. HumanMessage, AIMessage) to the history.
        
        Args:
            message (BaseMessage): The formatted message to add.
        """
        self._messages.append(message)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """
        Appends a sequence of BaseMessage objects to the history sequentially.
        
        Args:
            messages (list[BaseMessage]): A list of properly formatted messages.
        """
        for message in messages:
            self.add_message(message)

    def clear(self):
        """Clears all messages from the conversation history."""
        self._messages = []
    
    @property
    def messages(self) -> list[BaseMessage]:
        """
        Retrieves the entire stored list of messages.
        
        Returns:
            list[BaseMessage]: The chronological list of messages in this session.
        """
        return self._messages

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        """
        Overwrites the completely stored list of messages with a new list.
        
        Args:
            value (list[BaseMessage]): The new chronological list of messages.
        """
        self._messages = value

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or creates a chat message history for a specific session.

    Args:
        session_id (str): A unique identifier for the conversation session.

    Returns:
        BaseChatMessageHistory: The chat history object associated with the session.
    """
    if session_id not in AGENT_STORE:
        AGENT_STORE[session_id] = ChatMessageHistory()
    return AGENT_STORE[session_id]


def main():

    configure_environment(("OPENAI_API_KEY", "TAVILY_API_KEY"))

    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not openai_api_key:
        raise ValueError("OpenAI API key is not set")
    if not tavily_api_key:
        raise ValueError("Tavily API key is not set")
    
    tavily_search = TavilySearch(max_results=5)
    tools =[tavily_search]

    llm = get_llm(openai_api_key,temperature=0,tools=tools)
    prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Always introduce yourself and remember our past conversations."),
    ("placeholder", "{chat_history}"), # This is where the memory will be inserted
    ("user", "{input}")
    ])

    chain_with_history = prompt_with_history | llm

    conversational_chain  = RunnableWithMessageHistory(
        chain_with_history,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output"
    )
    print("Agent with memory initialized")
    config = create_session_config("user1", "conversation_session")

    while True:
        user_input = input("\n you:")
        if user_input.lower()  in ("exit","quit"):
            break

        response = conversational_chain.invoke({"input": user_input}, config=config)
        console.print(Panel(Markdown(response.content), title="AI Agent", border_style="cyan", box=ASCII))
    
    sys.exit(0)

if __name__ == "__main__":
    main()
