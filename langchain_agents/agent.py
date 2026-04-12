import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_agents.load_env import configure_environment
from langchain_agents.skills import review_code_tool, generate_code_tool

def get_dev_agent(model_name="gemini-2.0-flash"):
    """
    Returns a LangGraph React agent equipped with code review and generation tools.
    This meta-agent can help design, build, and optimize other agents.
    """
    try:
        configure_environment(["GOOGLE_API_KEY"])
    except ValueError as e:
        if not os.getenv("GOOGLE_API_KEY"):
            raise e

    llm = ChatGoogleGenerativeAI(model=model_name)
    tools = [review_code_tool, generate_code_tool]
    
    # Using LangGraph's prebuilt React agent
    agent = create_react_agent(llm, tools)
    return agent

if __name__ == "__main__":
    # Small demonstration of self-building capability
    agent = get_dev_agent()
    task = (
        "Project Requirement: Build a LangChain agent that can search the web and summarize results. "
        "Steps:\n"
        "1. Generate the Python code for this agent.\n"
        "2. Review the generated code to ensure it follows best practices."
    )
    
    print("\n" + "="*60)
    print("🚀 LANGCHAIN DEVELOPER AGENT")
    print("="*60)
    print(f"TASK: {task}\n")
    
    inputs = {"messages": [HumanMessage(content=task)]}
    
    try:
        for chunk in agent.stream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            if message.type == "ai":
                print(f"\n[AI]: {message.content}")
            elif hasattr(message, "tool_calls") and message.tool_calls:
                print(f"\n[TOOL CALL]: Calling {message.tool_calls[0]['name']}...")
    except Exception as e:
        print(f"Error during execution: {e}")
