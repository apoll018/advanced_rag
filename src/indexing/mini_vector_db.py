"""
Project: Advanced RAG Pipeline
Author: Lucas Henrique da S. Medeiros
GitHub: https://github.com/apoll018
Created: 2026

Este código é de autoria original. Uso sujeito à licença do projeto.
"""

"""
Módulo Mini-VectorDB (Router Semântico)
---------------------------------------
Gerencia um banco vetorial leve (em memória/pickle) usado para roteamento de documentos.
Funciona mapeando Termos-Chave -> Documentos (Normas).

"""

import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Import interno
from components.embeddings import get_embedding_function

# Configuração de Caminhos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CAMINHO_DB_PADRAO = os.getenv("MINI_VECT_DB_PATH", os.path.join(BASE_DIR, "data", "mini_vector_db.pkl"))

# ==========================================
# 💾 PERSISTÊNCIA E CRIAÇÃO
# ==========================================

def criar_mini_db(sinonimos: Dict[str, List[str]], caminho: str = CAMINHO_DB_PADRAO) -> Dict[str, Any]:
    """
    Cria um novo mini-DB vetorial a partir de um dicionário de termos.
    
    Args:
        sinonimos: Dict onde chave=NomeDoDoc e valor=ListaDeTermos.
        caminho: Local para salvar o arquivo .pkl.
    """
    emb = get_embedding_function()
    mini_db = {}

    print(f"[INIT] Criando Mini-DB para {len(sinonimos)} documentos...")

    for doc, termos in sinonimos.items():
        termos_limp = [t.strip() for t in termos if isinstance(t, str) and t.strip()]
        
        if not termos_limp:
            print(f"[WARN] Documento '{doc}' ignorado (sem termos válidos).")
            continue

        # Vetorização em lote (Batch Embedding)
        textos = [t for t in termos_limp]
        vetores = np.array(emb.embed_documents(textos))

        # Normalização L2 (para uso com Produto Escalar / Cosine Similarity)
        # Evita divisão por zero com + 1e-10
        normas = np.linalg.norm(vetores, axis=1, keepdims=True) + 1e-10
        vetores_normalizados = vetores / normas

        # Estrutura de armazenamento leve
        mini_db[doc] = [
            {"term": t, "vec": v} 
            for t, v in zip(termos_limp, vetores_normalizados)
        ]

    _salvar_pickle(mini_db, caminho)
    return mini_db

def atualizar_mini_db(doc: str, termos: List[str], caminho: str = CAMINHO_DB_PADRAO) -> None:
    """
    Adiciona ou atualiza os termos de um ÚNICO documento no banco existente.
    """
    # 1. Carrega ou Inicia Novo
    try:
        mini_db = carregar_mini_db(caminho)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        mini_db = {}

    termos_limp = [t.strip() for t in termos if isinstance(t, str) and t.strip()]

    if not termos_limp:
        if doc in mini_db:
            del mini_db[doc]
            print(f"[INFO] '{doc}' removido do Mini-DB (lista de termos vazia).")
        return

    # 2. Vetorização
    emb = get_embedding_function()
    textos = [t for t in termos_limp]
    vetores = np.array(emb.embed_documents(textos))
    
    # Normalização
    normas = np.linalg.norm(vetores, axis=1, keepdims=True) + 1e-10
    vetores_normalizados = vetores / normas

    # 3. Atualização (Sobrescreve entrada anterior)
    # Converte para float32 para economizar RAM
    mini_db[doc] = [
        {"term": t, "vec": v.astype(np.float32)} 
        for t, v in zip(termos_limp, vetores_normalizados)
    ]

    _salvar_pickle(mini_db, caminho)

def carregar_mini_db(caminho: str = CAMINHO_DB_PADRAO) -> Dict[str, Any]:
    """Carrega o banco do disco."""
    with open(caminho, "rb") as f:
        return pickle.load(f)

def _salvar_pickle(dados: Any, caminho: str) -> None:
    """Helper para salvar arquivo com segurança."""
    try:
        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
        with open(caminho, "wb") as f:
            pickle.dump(dados, f)
        # print(f"[OK] Mini-DB salvo em {caminho}") # Comentado para reduzir log
    except Exception as e:
        print(f"[ERROR] Falha ao salvar Mini-DB: {e}")

# ==========================================
# 🔍 LÓGICA DE CONSULTA (ROUTER)
# ==========================================

def consultar_mini_db(
    query: str,
    mini_db: Dict[str, List[Dict[str, Any]]],
    emb: Any,
    top_k_termos: int = 3,
    top_k_docs: int = 5,
    top_k_por_doc: int = 2,
    similarity_threshold: float = 0.55,
) -> Tuple[List[str], List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Realiza busca semântica no Mini-DB para identificar documentos candidatos.
    Usa suavização logarítmica (diminishing returns) para evitar que documentos
    longos dominem simplesmente por terem muitos termos.
    """

    # ----------------------------------------------
    # 1. Embedding da Query
    # ----------------------------------------------
    q_vec = np.array(emb.embed_query(query), dtype=np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    por_doc = defaultdict(list)

    # ----------------------------------------------
    # 2. Comparação Vetorial
    # ----------------------------------------------
    for norma, lista_termos in mini_db.items():
        for item in lista_termos:
            vec = item["vec"]
            sim = float(np.dot(q_vec, vec))  # Dot product == cosine (normalizado)

            if sim >= similarity_threshold:
                por_doc[norma].append((sim, item["term"]))

    # ----------------------------------------------
    # 3. Agregação com Suavização Logarítmica
    # ----------------------------------------------
    doc_scores = []

    for norma, sim_terms in por_doc.items():
        # Ordena pelos melhores matches
        sim_terms.sort(reverse=True, key=lambda x: x[0])

        # Pega somente os top N (top_k_por_doc)
        top_sims = [s[0] for s in sim_terms[:top_k_por_doc]]

        if top_sims:
            mean_score = np.mean(top_sims)     # Qualidade média dos melhores matches
            count = len(top_sims)             # Quantidade de matches usados

            # ----------------------------------------------------
            #  SUAVIZAÇÃO LOGARÍTMICA
            #
            # aumentar o score quando há mais matches,
            # mas sem explodir o valor (como multiplicar por N).
            #
            # log(count) cresce mais lento:
            # - count=1 => log=0
            # - count=2 => log=0.69
            # - count=3 => log=1.10
            #
            # E o "+1" garante que o multiplicador nunca seja < 1.
            # ----------------------------------------------------
            log_bonus = 1 + np.log(count)

            score = mean_score * log_bonus
            doc_scores.append((norma, score))

    # Ordena pelo score final
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs_scores = doc_scores[:top_k_docs]

    # ----------------------------------------------
    # 4. Extração dos Termos (Query Expansion)
    # ----------------------------------------------
    top_termos = []

    if top_docs_scores:
        doc_vencedor = top_docs_scores[0][0]

        if doc_vencedor in por_doc:
            matches = sorted(por_doc[doc_vencedor], key=lambda x: x[0], reverse=True)

            for _, term in matches:
                if term not in top_termos:
                    top_termos.append(term)
                if len(top_termos) >= top_k_termos:
                    break

    return top_termos, doc_scores, top_docs_scores


# ==========================================
# 🛠️ UTILITÁRIOS DE FILTRO
# ==========================================

def gerar_filtro_chroma(normas_identificadas: List[str]) -> Optional[Dict[str, Any]]:
    """
    Gera o dicionário de filtro ($in) para o ChromaDB.
    """
    if not normas_identificadas:
        return None

    # Filtra strings vazias ou muito curtas (ruído)
    normas_validas = [n for n in normas_identificadas if n and len(n) > 1]
    
    if not normas_validas:
        return None

    filtro = {
        "documento_completo": { 
            "$in": normas_validas
        }
    }
    print(f"[ROUTER] Filtro aplicado: {normas_validas}")
    return filtro