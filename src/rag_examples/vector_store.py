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

df = pd.read_csv("knowledge/reviews.csv")
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location = "./chroma_store_db"
add_documents = not os.path.exists(db_location)

if add_documents:
  documents = []
  ids = []

  for i, row in df.iterrows():
    document = Document(
      page_content = row["Title"] + " " + "row[Review]",
      metadata = { "rating": row["Rating"], "date": row["Data"] },
      id = str(i)
    )
    ids.append(str(i))
    documents.append(document)

vector_stroe = Chroma(
  collection_name="Restaurant Reviews",
  persist_directory=db_location,
  embedding_function=embeddings
)

if add_documents:
  vector_stroe.add_documents(documents=documents, ids=ids)

retriever = vector_stroe.as_retriever(
  search_kwargs={"k": 5}
)
