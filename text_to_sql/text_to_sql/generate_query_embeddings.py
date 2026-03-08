"""
Create embeddings and store them in ChromaDB.

Loads training data (dev.json), extracts evidence documents,
generates embeddings via Gemini, and stores them in a persistent collection.
"""
import logging
from typing import List, Optional

from langchain_core.documents import Document

from .settings import get_settings, Settings
from .common import (
    load_training_data,
    create_embeddings,
    create_chroma_client,
    create_vector_store,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document preparation
# ---------------------------------------------------------------------------

def prepare_documents(training_data: list) -> List[Document]:
    """Convert training examples into LangChain Documents."""
    documents: List[Document] = []
    for item in training_data:
        evidences = [e.strip() for e in item["evidence"].split(";") if e.strip()]
        for evidence in evidences:
            documents.append(
                Document(
                    page_content=evidence,
                    metadata={"db_id": item["db_id"]},
                )
            )
    logger.info(f"📄 Prepared {len(documents)} documents from training data.")
    return documents

def log_db_distribution(training_data: list) -> None:
    """Print how many examples exist per database."""
    counts: dict[str, int] = {}
    for item in training_data:
        db_id = item["db_id"]
        counts[db_id] = counts.get(db_id, 0) + 1

    logger.info("📊 Database distribution:")
    for db_id, count in sorted(counts.items()):
        logger.info(f"   {db_id}: {count} examples")

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    file_path: Optional[str] = None,
    vector_db_path: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: str = "documents",
    reset: bool = True,
    settings: Optional[Settings] = None,
) -> None:
    """End-to-end pipeline: load data → embed → store in ChromaDB."""
    cfg = settings or get_settings()
    file_path = file_path or str(cfg.FILE_PATH)
    vector_db_path = vector_db_path or str(cfg.VECTOR_DB_PATH)
    api_key = api_key or cfg.GOOGLE_API_KEY

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is required. Set it in .env, environment, "
            "or pass it directly via api_key parameter."
        )

    logger.info(f"🔧 Settings profile: {cfg.ENV}")
    logger.info(f"   FILE_PATH      = {file_path}")
    logger.info(f"   VECTOR_DB_PATH = {vector_db_path}")

    # 1. Load data
    training_data = load_training_data(file_path)
    if training_data is None:
        return

    # 2. Prepare documents
    documents = prepare_documents(training_data)
    if not documents:
        logger.warning("⚠️  No documents to store — nothing to do.")
        return

    # 3. Create embeddings + vector store
    embeddings = create_embeddings(api_key)
    chroma_client = create_chroma_client(vector_db_path)

    try:
        vector_store = create_vector_store(
            chroma_client, embeddings, collection_name, reset=reset
        )

        vector_store.add_documents(documents)
        logger.info(f"✅ Stored {len(documents)} documents in '{collection_name}'.")

        log_db_distribution(training_data)
    finally:
        chroma_client.close()
        logger.info("🔒 ChromaDB client closed.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    run()
