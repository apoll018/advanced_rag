# core_retrievers.py

from typing import Dict, Any
from langchain_community.retrievers import BM25Retriever


def setup_retrievers(state: Dict[str, Any], filtro_metadata: Dict[str, Any]):
    """
    Configura os retrievers SEM ParentDocumentRetriever.
    Agora o pipeline opera sobre chunks únicos (small).
    """

    # ========================================================
    # SEMÂNTICO — Chroma
    # ========================================================

    search_kwargs = {
        "k": 80,
    }

    # Filtro opcional
    if filtro_metadata:
        search_kwargs["filter"] = filtro_metadata

    retriever_semantico = state["vectorstore"].as_retriever(
        search_kwargs=search_kwargs
    )

    # ========================================================
    # LEXICAL — BM25
    # ========================================================

    retriever_bm25 = BM25Retriever.from_documents(state["documentos_bm25"])
    retriever_bm25.k = 30

    return retriever_semantico, retriever_bm25
