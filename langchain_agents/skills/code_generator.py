import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_agents.load_env import configure_environment

class CodeGenerator:
    """Skill to generate LangChain agent code."""
    
    def __init__(self, model_name="gemini-2.0-flash"):
        try:
            configure_environment(["GOOGLE_API_KEY"])
        except ValueError as e:
            # Fallback for environments where keys might be set differently
            if not os.getenv("GOOGLE_API_KEY"):
                raise e

        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Python developer specializing in LangChain and Agentic AI. "
                       "Your task is to generate high-quality, production-ready Python code for a LangChain agent "
                       "based on the user's prompt. Use best practices like tool definitions, "
                       "clean memory management, and robust error handling. "
                       "Wrap the code in triple backticks with 'python' language identifier."),
            ("user", "Generate a LangChain agent for the following task:\n\n{task}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, task: str) -> str:
        """Generates code based on the task prompt."""
        if not task.strip():
            return "Error: No task provided for code generation."
        return self.chain.invoke({"task": task})

@tool
def generate_code_tool(task: str) -> str:
    """
    Generates Python code for a LangChain agent based on a natural language task description.
    Use this tool when you need to write or scaffold new agent logic.
    Input should be a clear description of what the agent should do.
    """
    generator = CodeGenerator()
    return generator.generate(task)
