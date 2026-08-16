"""State definitions and Pydantic schemas for Smart Query Router."""

from typing import List, Literal, Optional, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class RouterState(TypedDict, total=False):
    """LangGraph state model for Smart Query Router workflow.

    Attributes:
        query: User input query.
        intent: Classified intent ("search" or "direct").
        reasoning: Justification for intent classification.
        search_results: Retrieved web search payload or summary.
        response: Final generated response to the user.
    """
    query: str
    intent: Optional[Literal["search", "direct"]]
    reasoning: Optional[str]
    search_results: Optional[Union[List[dict], str]]
    response: Optional[str]


class IntentClassification(BaseModel):
    """Pydantic model for structured LLM intent classification output."""

    intent: Literal["search", "direct"] = Field(
        ...,
        description=(
            "Classified intent: 'search' if query requires real-time, live, current facts/news/weather/stock info, "
            "or 'direct' if it can be answered using general internal model knowledge."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why web search is or is not required.",
    )
