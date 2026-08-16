"""Prompt definitions and prompt_helper integrations for Smart Query Router."""

from prompt_helper import PromptBuilder

INTENT_DETECTION_SYSTEM = """You are an expert query classification agent.
Your sole job is to determine whether a user query requires real-time/current web search information ("search") or can be answered directly from general internal knowledge ("direct").

Instructions:
1. Classify as "search" if the query asks about:
   - Current events, breaking news, or recent developments
   - Real-time weather, stock prices, live sports scores, or schedules
   - Specific recent publications, release dates, or status updates
   - Explicit requests for web search or current data

2. Classify as "direct" if the query asks about:
   - Historical facts, scientific principles, or general knowledge
   - Math equations, logic puzzles, coding syntax, or refactoring
   - Definitions, creative writing, translation, or static explanations

Provide your classification as structured output containing 'intent' ("search" or "direct") and 'reasoning'."""

INTENT_DETECTION_HUMAN = """User Query: {query}

Determine the intent for routing."""


DIRECT_RESPONSE_SYSTEM = """You are a knowledgeable and helpful AI assistant.
Answer the user's question directly, clearly, and accurately using your internal knowledge."""

DIRECT_RESPONSE_HUMAN = """User Query: {query}"""


WEB_SEARCH_RESPONSE_SYSTEM = """You are a helpful AI assistant augmented with real-time web search results.
Answer the user's query using the provided search context. Be factual, concise, and synthesize information cleanly."""

WEB_SEARCH_RESPONSE_HUMAN = """User Query: {query}

Web Search Results Context:
{search_context}

Provide a well-structured and complete answer based on the search results context above."""


def build_intent_detection_messages(query: str):
    """Build messages list for intent detection node using prompt_helper PromptBuilder."""
    return (
        PromptBuilder("intent_detection")
        .system(INTENT_DETECTION_SYSTEM)
        .human(INTENT_DETECTION_HUMAN)
        .build(query=query)
    )


def build_direct_response_messages(query: str):
    """Build messages list for direct response node using prompt_helper PromptBuilder."""
    return (
        PromptBuilder("direct_response")
        .system(DIRECT_RESPONSE_SYSTEM)
        .human(DIRECT_RESPONSE_HUMAN)
        .build(query=query)
    )


def build_web_search_response_messages(query: str, search_context: str):
    """Build messages list for web search node using prompt_helper PromptBuilder."""
    return (
        PromptBuilder("web_search_response")
        .system(WEB_SEARCH_RESPONSE_SYSTEM)
        .human(WEB_SEARCH_RESPONSE_HUMAN)
        .build(query=query, search_context=search_context)
    )
