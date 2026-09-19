from typing import TypedDict,List,Annotated
from operator import add
from langgraph.graph import StateGraph, START ,END
from langgraph.types import Send

class MasterState(TypedDict):

    topic:str
    subtopics:List[str]
    research_results:Annotated[List[str],add]
    final_report:str

class ResearchState(TypedDict):

    subtopic:str
    research_results:List[str]


def generate_subtopics(state:MasterState):
    topic = state["topic"]
    subtopics = [
        f'{topic} - Explain',
        f'{topic} - Current Trends',
        f'{topic} - Future',
    ]
    return { "subtopics":subtopics}

def research_subtopic(state:ResearchState):
    subtopic = state["subtopic"]
    result  = f"Research result is "
    return { "research_results" :[result]}

def compile_report(state:MasterState):
    research_results = state["research_results"]
    final_report = "-"*50 + "\n"
    final_report += "COMPREHENSIVE REPORT \n "
    final_report = "-"*50 + "\n"
    for i, result in enumerate(research_results,1):
        final_report += f'{i}. {result} \n\n'
    return { "final_report" :final_report}

def continue_research(state:MasterState):
    return [Send("research_subtopic",{"subtopic":subtopic})
             for subtopic in state["subtopics"]
                ]

builder = StateGraph(MasterState)
builder.add_node(generate_subtopics)
builder.add_node(research_subtopic)
builder.add_node(compile_report)
builder.add_edge(START, "generate_subtopics")
builder.add_edge("generate_subtopics", continue_research)
builder.add_edge("research_subtopic", "compile_report")
builder.add_edge("compile_report",END)
