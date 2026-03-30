"""
Módulo Retrieval Utils
----------------------
Utilitários matemáticos e lógicos para manipulação de resultados de busca.

Funções:
- aplicar_rrf: Combina múltiplos rankings usando Reciprocal Rank Fusion.
- gerar_filtro_chroma: Cria dicionários de filtro compatíveis com a sintaxe do ChromaDB.
"""

from collections import defaultdict
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# ==============================================================
# === COMBINAÇÃO DE RANKINGS (RRF) ===
# ==============================================================

def aplicar_rrf(resultados: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Combina listas de documentos de diferentes retrievers usando Reciprocal Rank Fusion.
    
    A fórmula de pontuação é: score += 1 / (k + rank).
    Prioriza documentos que aparecem no topo de múltiplas listas.
    
    Args:
        resultados: Lista de listas de documentos (ex: [docs_bm25, docs_vetorial]).
        k: Constante de suavização (padrão 60).
        
    Returns:
        Lista única de documentos, reordenada pelo score RRF, sem duplicatas.
    """
    scores = defaultdict(lambda: {"doc": None, "score": 0.0})

    for lista_docs in resultados:
        for rank, doc in enumerate(lista_docs):
            
            # Identificação única do documento (Parent ID > Doc ID)
            # Isso garante que chunks diferentes do mesmo pai somem score para o pai
            doc_id = doc.metadata.get("parent_doc_id") or doc.metadata.get("doc_id") or str(doc)

            # Inicializa se for a primeira vez que vemos este doc
            if scores[doc_id]["doc"] is None:
                scores[doc_id]["doc"] = doc

            # Acumula pontuação RRF
            scores[doc_id]["score"] += 1 / (k + rank)

    # Ordena descrescentemente pelo score acumulado
    docs_ordenados = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    
    # Reconstrói a lista final
    lista_final = [item["doc"] for item in docs_ordenados]

    print(f"[RRF] Fusão concluída. {len(lista_final)} documentos únicos combinados.")
    return lista_final
