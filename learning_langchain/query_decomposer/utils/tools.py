import ast
import operator
import time
import random
import wikipedia
from typing import Any,List
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Create singleton instance of DuckDuckGoSeachRun
_instance = None


def pick_tool(tool_hint: str | None, sub_question: str,keywords,tools) -> str:
    """Choose the best tool for a sub-question."""
    if tool_hint:
        hint = tool_hint.strip().lower().replace(" ", "_")
        if hint in tools:
            return hint
    # Heuristic fallback
    if keywords.search(sub_question):
        return "wikipedia_lookup"
    return "web_search"


def get_search_instance():

    global _instance
    if _instance is None:
        _instance = DuckDuckGoSearchRun()
    return _instance

class Websearch():

    def __init__(self) -> None:
        self.search_instance = get_search_instance()
    
    def __call__(self,query:str):
        try:
            results = self.search_instance.run(query)
            return results
        except Exception as e:
            return f"Search failed: {e}"

class WikiSearch():

    def __init__(self,sentences:int=5) -> None:
        self.sentences = sentences
    
    def __call__(self,topic:str):
        try:
            return wikipedia.summary(topic, sentences=self.sentences, auto_suggest=True)
        except wikipedia.exceptions.DisambiguationError as e:
            if not e.options:
                return f"Wikipedia lookup for '{topic}' was ambiguous with no options."
            try:
                return wikipedia.summary(e.options[0], sentences=self.sentences, auto_suggest=False)
            except Exception as inner_e:
                return f"Wikipedia lookup failed for '{e.options[0]}': {inner_e}"
        except wikipedia.exceptions.PageError:
            return f"Wikipedia page not found: '{topic}'. Try a web search instead."
        except Exception as e:
            return f"Wikipedia lookup failed: {e}"
    
def _evaluate_ast(node):
    operators_map = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        return operators_map[type(node.op)](
            _evaluate_ast(node.left), _evaluate_ast(node.right)
        )
    elif isinstance(node, ast.UnaryOp):
        return operators_map[type(node.op)](_evaluate_ast(node.operand))
    else:
        raise TypeError(f"Unsupported node type: {type(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. E.g. '(1.4 / 1.2 - 1) * 100' for percentage diff."""
    try:
        if len(expression) > 200:
            return "Error: Expression too long."
        tree = ast.parse(expression.strip(), mode="eval")
        result = _evaluate_ast(tree.body)
        return str(round(result, 6)) if isinstance(result, float) else str(result)
    except SyntaxError:
        return "Calculation error: Invalid syntax."
    except (KeyError, TypeError) as e:
        return f"Calculation error: {e}"
    except Exception as e:
        return f"Calculation error: {e}"




# Instantiate the tools from utils
_web_search_inst = Websearch()
_wiki_search_inst = WikiSearch()

@tool
def web_search(query: str) -> str:
    """Search the web for current events, news, and general queries using DuckDuckGo."""
    return _web_search_inst(query)

@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up encyclopedic facts, history, biographies, and science on Wikipedia."""
    return _wiki_search_inst(topic)