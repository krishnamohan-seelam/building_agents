"""
Example: 2

This module provides a tutorial on building a conversational AI agent with memory 
using LangChain and OpenAI models. It demonstrates how to integrate `RunnableWithMessageHistory`
to manage session-based chat history, allowing the LLM to recall previous interactions within
a given session ID. Additionally, it highlights using the `rich` library to present the 
AI's responses in an aesthetically pleasing way in the terminal.
"""
import os
from load_env import configure_environment

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage,BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.box import ASCII

# Set environment variable to force UTF-8 before Console is fully bound
os.environ["PYTHONIOENCODING"] = "utf-8"
console = Console()


AGENT_STORE = dict()

def get_llm(apikey, model="gpt-4o", temperature=0):
    """
    Initializes and returns a ChatOpenAI language model instance.

    Args:
        apikey (str): The OpenAI API key used for authentication.
        model (str, optional): The model version to use. Defaults to "gpt-4o".
        temperature (float, optional): Sampling temperature. Defaults to 0 for deterministic outputs.

    Returns:
        ChatOpenAI: A configured LangChain ChatOpenAI LLM object.
    """
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
    """
    Main entry point for running the agentic memory tutorial.
    
    This function configures the environment, instantiates the OpenAI LLM, creates a 
    conversational prompt with a history placeholder, and builds a runnable chain 
    that binds session history. It then runs several test interactions to demonstrate 
    that the AI remembers context.
    """

    configure_environment()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OpenAI API key is not set")

    llm = get_llm(openai_api_key)
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
    config = {"configurable": {"session_id": "conversation_session"}}

    response = conversational_chain.invoke({"input": "Hi, my name is Krishna"}, config=config)
    console.print(Panel(Markdown(response.content), title="AI Agent", border_style="cyan", box=ASCII))

    response_1 = conversational_chain.invoke({"input": "What is my name?"}, config=config)
    console.print(Panel(Markdown(response_1.content), title="AI Agent", border_style="cyan", box=ASCII))

    response_2 = conversational_chain.invoke({"input": "Tell me fun fact about Python programming language?"}, config=config)
    console.print(Panel(Markdown(response_2.content), title="AI Agent", border_style="cyan", box=ASCII))

    response_3 = conversational_chain.invoke({"input": "Explain me the meaning of my name?"}, config=config)
    console.print(Panel(Markdown(response_3.content), title="AI Agent", border_style="cyan", box=ASCII))

if __name__ == "__main__":
    main()