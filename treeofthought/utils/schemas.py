from pydantic import BaseModel, Field


class ThoughtCandidate(BaseModel):
    thought: str = Field(description="The next reasoning step, self-contained.")


class GeneratorOutput(BaseModel):
    candidates: list[ThoughtCandidate] = Field(
        description="k distinct next-step thoughts, no duplicates."
    )


class EvaluatorOutput(BaseModel):
    score: float = Field(ge=0, le=10, description="Heuristic quality score, 0-10.")
    reason: str = Field(description="One-sentence justification for the score.")


class SynthesisOutput(BaseModel):
    final_answer: str = Field(
        description="Complete, well-formatted answer to the problem."
    )
