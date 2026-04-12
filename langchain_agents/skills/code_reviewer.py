import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_agents.load_env import configure_environment

class CodeReviewer:
    """Skill to review LangChain agent code."""
    
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
                       "Your task is to review the provided code and provide constructive feedback, "
                       "identifying bugs, performance issues, and best practice violations. "
                       "Always provide the review in a professional, premium-designed Markdown format."),
            ("user", "Review the following LangChain agent code:\n\n{code}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def review(self, code: str) -> str:
        """Reviews the provided code string."""
        if not code.strip():
            return "Error: No code provided for review."
        return self.chain.invoke({"code": code})

@tool
def review_code_tool(code: str) -> str:
    """
    Analyzes Python code for LangChain agents. 
    Use this tool to find bugs, performance issues, and improve code quality.
    Input should be the raw Python code string.
    """
    reviewer = CodeReviewer()
    return reviewer.review(code)
