"""Node implementations for Smart Query Router state graph."""

import os
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from smart_query_router.state import RouterState, IntentClassification
from smart_query_router.prompts import (
    build_intent_detection_messages,
    build_direct_response_messages,
    build_web_search_response_messages,
)

logger = logging.getLogger(__name__)


def _get_default_llm():
    """Retrieve default OpenAI LLM instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable is not set. Using fallback key placeholder for initialization.")
        api_key = "placeholder_key"
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


def _get_default_search_tool():
    """Retrieve default Tavily search tool instance or fallback tool."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY environment variable is not set. Using fallback key placeholder for initialization.")
        api_key = "placeholder_key"
    return TavilySearch(max_results=3, tavily_api_key=api_key)



def format_search_results(results: Union[List[dict], str, Any]) -> str:
    """Format raw search results into a clean text context string."""
    if isinstance(results, str):
        return results
    if isinstance(results, list):
        formatted = []
        for i, item in enumerate(results, 1):
            if isinstance(item, dict):
                title = item.get("title", f"Result {i}")
                content = item.get("content", item.get("snippet", str(item)))
                url = item.get("url", "")
                formatted.append(f"[{i}] {title}\nURL: {url}\nContent: {content}")
            else:
                formatted.append(f"[{i}] {str(item)}")
        return "\n\n".join(formatted) if formatted else "No relevant search results found."
    return str(results)


def intent_detection_node(
    state: RouterState,
    config: Optional[RunnableConfig] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Node 1: Detect query intent using OpenAI structured output.

    Args:
        state: Current router state containing user 'query'.
        config: Optional LangGraph/LangChain RunnableConfig containing runtime configurable settings.
        llm: Optional LLM instance.

    Returns:
        Dictionary with updated 'intent' and 'reasoning'.
    """
    query = state.get("query", "")
    logger.info("Executing intent_detection_node for query: '%s'", query)

    if llm is None and config is not None:
        llm = config.get("configurable", {}).get("llm")

    if llm is None:
        llm = _get_default_llm()

    structured_llm = llm.with_structured_output(IntentClassification)
    messages = build_intent_detection_messages(query)

    classification: IntentClassification = structured_llm.invoke(messages)

    logger.info(
        "Query classified as '%s' (Reasoning: %s)",
        classification.intent,
        classification.reasoning,
    )

    return {
        "intent": classification.intent,
        "reasoning": classification.reasoning,
    }


def web_search_node(
    state: RouterState,
    config: Optional[RunnableConfig] = None,
    llm: Optional[Any] = None,
    search_tool: Optional[Any] = None,
) -> Dict[str, Any]:
    """Node 2: Execute web search and generate augmented answer.

    Args:
        state: Current router state containing user 'query'.
        config: Optional LangGraph/LangChain RunnableConfig containing runtime configurable settings.
        llm: Optional LLM instance.
        search_tool: Optional search tool instance (e.g. TavilySearchResults).

    Returns:
        Dictionary with 'search_results' and 'response'.
    """
    query = state.get("query", "")
    logger.info("Executing web_search_node for query: '%s'", query)

    configurable = config.get("configurable", {}) if config else {}

    if search_tool is None:
        search_tool = configurable.get("search_tool")
    if search_tool is None:
        search_tool = _get_default_search_tool()

    try:
        raw_results = search_tool.invoke({"query": query})
    except Exception as exc:
        logger.error("Web search tool invocation failed: %s", exc)
        raw_results = f"Search execution failed: {exc}"

    search_context = format_search_results(raw_results)

    if llm is None:
        llm = configurable.get("llm")
    if llm is None:
        llm = _get_default_llm()

    messages = build_web_search_response_messages(query, search_context)
    result = llm.invoke(messages)
    response_text = result.content if hasattr(result, "content") else str(result)

    return {
        "search_results": raw_results,
        "response": response_text,
    }


def direct_response_node(
    state: RouterState,
    config: Optional[RunnableConfig] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Node 3: Generate direct answer from internal LLM knowledge.

    Args:
        state: Current router state containing user 'query'.
        config: Optional LangGraph/LangChain RunnableConfig containing runtime configurable settings.
        llm: Optional LLM instance.

    Returns:
        Dictionary with updated 'response'.
    """
    query = state.get("query", "")
    logger.info("Executing direct_response_node for query: '%s'", query)

    if llm is None and config is not None:
        llm = config.get("configurable", {}).get("llm")

    if llm is None:
        llm = _get_default_llm()

    messages = build_direct_response_messages(query)
    result = llm.invoke(messages)
    response_text = result.content if hasattr(result, "content") else str(result)

    return {
        "response": response_text,
    }

