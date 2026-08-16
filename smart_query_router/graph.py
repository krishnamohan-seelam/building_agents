"""Graph compilation and routing logic for Smart Query Router."""

import logging
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from smart_query_router.state import RouterState
from smart_query_router.nodes import (
    intent_detection_node,
    web_search_node,
    direct_response_node,
)

logger = logging.getLogger(__name__)


def route_intent(state: RouterState) -> str:
    """Conditional edge router function.

    Evaluates state["intent"] and determines the next node in the graph.

    Args:
        state: RouterState containing classified 'intent'.

    Returns:
        Target node key: "web_search_node" or "direct_response_node".
    """
    intent = state.get("intent")
    logger.info("route_intent evaluating intent: '%s'", intent)

    if intent == "search":
        return "web_search_node"
    return "direct_response_node"


def build_router_graph(
    llm: Optional[Any] = None,
    search_tool: Optional[Any] = None,
) -> CompiledStateGraph:
    """Assemble and compile the Smart Query Router LangGraph StateGraph.

    Graph Architecture:
        START ──> intent_detection ──> [route_intent]
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
               web_search                                    direct_response
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                                           END

    Args:
        llm: Optional custom LLM instance for testing/dependency injection.
        search_tool: Optional custom search tool for testing/dependency injection.

    Returns:
        Compiled LangGraph StateGraph instance.
    """
    workflow = StateGraph(RouterState)

    # Add Nodes directly using top-level node functions
    workflow.add_node("intent_detection", intent_detection_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("direct_response", direct_response_node)

    # Add Entry Edge
    workflow.add_edge(START, "intent_detection")

    # Add Conditional Edge
    workflow.add_conditional_edges(
        "intent_detection",
        route_intent,
        {
            "web_search_node": "web_search",
            "direct_response_node": "direct_response",
        },
    )

    # Add Exit Edges
    workflow.add_edge("web_search", END)
    workflow.add_edge("direct_response", END)

    compiled_graph = workflow.compile()

    # If runtime dependencies were explicitly provided, bind them via configurable options
    configurable = {}
    if llm is not None:
        configurable["llm"] = llm
    if search_tool is not None:
        configurable["search_tool"] = search_tool

    if configurable:
        return compiled_graph.with_config(configurable=configurable)

    return compiled_graph

