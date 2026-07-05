"""
Text Summarisation Agent using a Map-Reduce LangChain pipeline.

Usage:
    python -m text_summarization.summarizer --filename path/to/document.txt
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import TokenTextSplitter

from text_summarization.utils import read_file

# ---------------------------------------------------------------------------
# Module-level logger — configured by the application entry-point (main)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 3_000
CHUNK_OVERLAP: int = 100

MAP_PROMPT_TEMPLATE: str = """\
You are a helpful assistant that summarizes text.
Summarize the following text in a concise manner that includes the main details.
Text: {chunk}"""

REDUCE_PROMPT_TEMPLATE: str = """\
You are a helpful assistant that summarizes text.
You are given several individual summaries. Produce a single, concise summary
that captures the main details from all of them.
Summaries: {summaries}"""

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _get_required_env(name: str) -> str:
    """Return the value of *name* from the environment, or raise clearly.

    Args:
        name: The environment variable name.

    Returns:
        The non-empty string value of the variable.

    Raises:
        EnvironmentError: If the variable is absent or empty.
    """
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Check your .env file."
        )
    return value


def _split_into_chunks(text: str) -> list[dict[str, str]]:
    """Split *text* into token-limited chunks for the map stage.

    Args:
        text: The raw document text to split.

    Returns:
        A list of dicts, each with a ``chunk`` key, ready for the map prompt.
    """
    return [{"chunk": chunk} for chunk in _splitter.split_text(text)]


def _join_summaries(items: list[dict[str, str]]) -> dict[str, str]:
    """Merge per-chunk summaries into a single string for the reduce stage.

    Args:
        items: List of dicts produced by the map stage, each containing
               a ``summary`` key.

    Returns:
        A dict with a ``summaries`` key containing all summaries joined by
        newlines, ready for the reduce prompt.
    """
    return {"summaries": "\n".join(item["summary"] for item in items)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the Google Generative AI LLM from environment config.

    Returns:
        A configured :class:`ChatGoogleGenerativeAI` instance.

    Raises:
        EnvironmentError: If ``MODEL`` or ``GOOGLE_API_KEY`` are not set.
    """
    return ChatGoogleGenerativeAI(
        model=_get_required_env("MODEL"),
        temperature=0,
        api_key=_get_required_env("GOOGLE_API_KEY"),
    )


def run(filename: Path) -> str:
    """Summarise the text file at *filename* using a Map-Reduce pipeline.

    Args:
        filename: Path to the plain-text file to summarise.

    Returns:
        The final summary string produced by the reduce stage.

    Raises:
        FileNotFoundError: If *filename* does not exist.
        EnvironmentError: If required environment variables are missing.
    """
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    logger.info("Reading file: %s", filename)
    content = read_file(filename)
    logger.info("File content read successfully: %d characters.", len(content))

    llm = load_llm()

    # -- Map stage: split into chunks -----------------------------------------
    text_chunks_chain = RunnableLambda(_split_into_chunks)

    map_prompt = PromptTemplate.from_template(MAP_PROMPT_TEMPLATE)
    summarize_chunk_chain = map_prompt | llm | StrOutputParser()
    summarize_map_chain = RunnableParallel({"summary": summarize_chunk_chain})

    # -- Reduce stage: join chunk summaries -----------------------------------
    reduce_prompt = PromptTemplate.from_template(REDUCE_PROMPT_TEMPLATE)
    summarize_reduce_chain = (
        RunnableLambda(_join_summaries)
        | reduce_prompt
        | llm
        | StrOutputParser()
    )

    # -- Full Map-Reduce chain ------------------------------------------------
    map_reduce_chain = text_chunks_chain | summarize_map_chain.map() | summarize_reduce_chain

    logger.info("Running Map-Reduce summarisation pipeline…")
    summary: str = map_reduce_chain.invoke(content)
    logger.info("Summarisation complete. Output: %d characters.", len(summary))
    return summary


def main() -> None:
    """CLI entry-point: parse arguments, load config, run the summariser."""
    # load_dotenv MUST run before logging setup so that any LOG_LEVEL env var
    # (or other config) is available when basicConfig is called.
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    logger.info("Starting summarizer agent…")

    parser = argparse.ArgumentParser(description="Map-Reduce text summarizer")
    parser.add_argument(
        "--filename",
        required=True,
        help="Path to the plain-text file to summarise",
    )
    args = parser.parse_args()
    filename = Path(args.filename)

    try:
        summary = run(filename)
        print(summary)
        logger.info("Summarizer agent finished.")
    except FileNotFoundError as exc:
        logger.error("Input file not found: %s", exc)
        raise SystemExit(1) from exc
    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001 — catch-all at boundary only
        logger.exception("Unexpected error during summarisation: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()