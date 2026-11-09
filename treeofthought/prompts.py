from prompt_helper import PromptBuilder

GENERATOR_SYSTEM = """You are a creative problem-solving agent exploring a tree of
reasoning paths to solve a problem. Given the problem and the reasoning path so far,
propose {k} distinct, plausible next steps. Each step should:
  - Follow logically from the path so far (or start fresh if the path is empty).
  - Be meaningfully different from the other candidates you propose (no near-duplicates).
  - Be a single concrete step, not a full solution.
"""

GENERATOR_HUMAN_TEMPLATE = """Problem: {problem}

Reasoning path so far:
{path_so_far}

Propose {k} distinct next steps."""


EVALUATOR_SYSTEM = """You are a critical evaluator scoring one reasoning path toward
solving a problem. Score how promising this path is on a scale of 0-10, where:
  - 0-3: flawed, contradicts the problem, or a dead end.
  - 4-6: plausible but incomplete or unremarkable.
  - 7-10: correct so far and clearly converging on a strong final answer.
Judge only what's written — do not reward length or confidence.
"""

EVALUATOR_HUMAN_TEMPLATE = """Problem: {problem}

Reasoning path to evaluate:
{path}

Give a score and a one-sentence reason."""


SYNTHESIZER_PROMPT = """You are a synthesis agent. A tree-of-thought search has found
its best reasoning path for the problem below. Turn it into a complete, well-formatted
final answer.

Problem: {problem}

Winning reasoning path:
{winning_path}

Write the final answer, directly addressing the problem, using this path's reasoning."""


def build_generator_messages(problem: str, path_so_far: str, k: int):
    """Build generator messages """
    return (
        PromptBuilder("decomposer")
        .system(GENERATOR_SYSTEM)
        .human(GENERATOR_HUMAN_TEMPLATE)
        .build(problem=problem, path_so_far=path_so_far, k=k)
    )


def build_evaluator_messages(problem: str, path: str):
    return (
        PromptBuilder("evaluator")
        .system(EVALUATOR_SYSTEM)
        .human(EVALUATOR_HUMAN_TEMPLATE)
        .build(problem=problem, path=path)
    )


def build_synthesizer_messages(problem: str, winning_path: str):
    return (
        PromptBuilder("synthesizer")
        .human(SYNTHESIZER_PROMPT)
        .build(problem=problem, winning_path=winning_path)
    )
