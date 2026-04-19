import sys
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from langchain_tavily import TavilySearch
from tavily import TavilyClient
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
app = Server("web-search-mcp")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="web_search",
            description="Use this tool ONLY if the user explicitly asks to 'search using web'. Performs a web search and returns results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "web_search":
        # Print progress to stderr so we don't break MCP stdio protocol
        print(f"[MCP] Executing web_search with query: {arguments['query']}", file=sys.stderr)
        res = tavily_client.search(arguments["query"], max_results=2)
        return [types.TextContent(type="text", text=str(res))]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
