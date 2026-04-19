import os
import sys
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.box import ASCII

from mcp_client import mcp_environment
from agent_core import ainvoke_agent

os.environ["PYTHONIOENCODING"] = "utf-8"
console = Console()

async def interactive_loop(tools):
    print("Agent initialized with MCP capability.")
    while True:
        try:
            user_input = input("\n you: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.lower() in ("exit", "quit"):
            break
            
        response = await ainvoke_agent("cli_user", "session_1", user_input, tools=tools)
        console.print(Panel(Markdown(response), title="AI Agent", border_style="cyan", box=ASCII))

async def main():
    # Attempt to load environment vars from parent
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from load_env import configure_environment
        configure_environment(("GOOGLE_API_KEY", "TAVILY_API_KEY"))
    except ImportError:
        pass # If load_env is not found, assume dotenv
        from dotenv import load_dotenv
        load_dotenv()
        
    async with mcp_environment() as mcp_tools:
        await interactive_loop(mcp_tools)

if __name__ == "__main__":
    asyncio.run(main())
