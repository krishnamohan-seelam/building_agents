import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from mcp_client import mcp_environment
from agent_core import ainvoke_agent

# We hold tools globally for the lifetime of the application
mcp_tools_ref = []

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        mcp_tools_ref.extend(mcp_tools)
        yield
        mcp_tools_ref.clear()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    response = await ainvoke_agent(
        user_id=request.user_id,
        session_id=request.session_id,
        message=request.message,
        tools=mcp_tools_ref
    )
    return {"reply": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
