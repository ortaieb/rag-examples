"""
  Simple implementation of Embedded Chroma vector store.
  This implementation has a persistent store. In this first example it will check of the chroma-store
  exists on the filesystem and load only if the store is missing.

  Note:
  - The source files inside `<root>/knowledge` exepected to be immutable.
"""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd


def create_vector_store(db_location: str, embedding_model: str, k_value: int):
    """
    Create and configure vector store with given parameters.

    Args:
        db_location: Path where the Chroma database will be stored
        embedding_model: Ollama model to use for embeddings
        k_value: Number of documents to retrieve

    Returns:
        Chroma retriever instance
    """
    df = pd.read_csv("knowledge/reviews.csv")
    embeddings = OllamaEmbeddings(model=embedding_model)

    add_documents = not os.path.exists(db_location)

    if add_documents:
        documents = []
        ids = []

        for i, row in df.iterrows():
            document = Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)

    vector_store = Chroma(
        collection_name="Restaurant_Reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store.as_retriever(search_kwargs={"k": k_value})


# Default retriever for backward compatibility
try:
    from .config import CONFIG
    retriever = create_vector_store(
        CONFIG.vector_store.db_location,
        CONFIG.ollama.embedding_model,
        CONFIG.vector_store.k_value
    )
except ImportError:
    # When running as script, use absolute imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag_examples.config import CONFIG
    retriever = create_vector_store(
        CONFIG.vector_store.db_location,
        CONFIG.ollama.embedding_model,
        CONFIG.vector_store.k_value
    )
