"""
Boston Aerospace AI
OpenAI RAG System

Uses:
- OpenAI Embeddings
- ChromaDB
- OpenAI API

No Ollama dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from openai import OpenAI

from Config.Settings import (
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
)


class OpenAIRAG:

    def __init__(self):

        if not OPENAI_API_KEY:

            raise RuntimeError(
                "OPENAI_API_KEY não configurada."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        self.embed_model = OPENAI_EMBED_MODEL

        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = (
            self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def create_embedding(
        self,
        text: str,
    ) -> List[float]:

        response = self.client.embeddings.create(
            model=self.embed_model,
            input=text,
        )

        return response.data[0].embedding

    # ========================================================
    # INDEX DOCUMENT
    # ========================================================

    def add_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:

        embedding = self.create_embedding(
            text
        )

        self.collection.upsert(
            ids=[
                document_id
            ],
            documents=[
                text
            ],
            embeddings=[
                embedding
            ],
            metadatas=[
                metadata or {}
            ],
        )

        return True

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:

        embedding = self.create_embedding(
            query
        )

        results = self.collection.query(
            query_embeddings=[
                embedding
            ],
            n_results=n_results,
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        output = []

        for index, document in enumerate(
            documents
        ):

            output.append(
                {
                    "text": document,
                    "metadata": (
                        metadatas[index]
                        if index < len(metadatas)
                        else {}
                    ),
                }
            )

        return output

    # ========================================================
    # CONTEXT
    # ========================================================

    def build_context(
        self,
        query: str,
        n_results: int = 5,
    ) -> str:

        results = self.search(
            query=query,
            n_results=n_results,
        )

        if not results:

            return ""

        blocks = []

        for item in results:

            blocks.append(
                item["text"]
            )

        return "\n\n---\n\n".join(
            blocks
        )


# ============================================================
# SINGLETON
# ============================================================

_rag = None


def get_rag() -> OpenAIRAG:

    global _rag

    if _rag is None:

        _rag = OpenAIRAG()

    return _rag


def search_knowledge(
    query: str,
    n_results: int = 5,
) -> List[Dict[str, Any]]:

    return get_rag().search(
        query=query,
        n_results=n_results,
    )


def get_context(
    query: str,
    n_results: int = 5,
) -> str:

    return get_rag().build_context(
        query=query,
        n_results=n_results,
        )
