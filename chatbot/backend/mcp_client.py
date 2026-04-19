import os
import sys
import asyncio
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# We are using python to run our custom MCP server for web search
DEFAULT_MCP_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_mcp_server.py")],
    env={**os.environ}
)

@asynccontextmanager
async def mcp_environment(server_params: StdioServerParameters = DEFAULT_MCP_PARAMS):
    print("Initializing MCP Client session with", server_params.command, server_params.args)
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Load all the server exposed capabilities as Langchain Tools
                tools = await load_mcp_tools(session)
                yield tools
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        # fallback to no tools
        yield []
