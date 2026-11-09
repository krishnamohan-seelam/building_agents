from typing import List, Annotated, TypedDict, Literal

class Thought(TypedDict):
    id: str
    parent_id: str | None
    path: List[str]
    content: str
    score: float
    status: Literal["active", "pruned", "solved"]


def format_path(thought):
    "Concatenates the Thought paths" 
    if not thought["path"]:
        return  ""

    lines = []
    for i, step in enumerate(thought["path"],start=1):
        lines.append(f" Step{i}: {step}")
    return "\n".join(lines)

def new_id(parent_id,sibling_index):
    if parent_id is None:
        return "root"
    return f"{parent_id}.{sibling_index}"
