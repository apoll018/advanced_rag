import os
import time
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- Imports Internos ---
# (Removidos imports de DocStore e Splitters de Parent que não usamos mais)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Seus componentes existentes
from components.embeddings import get_embedding_function
from components.query_rewriter import preparar_consulta
from components.reranker import FlashRankReranker  
from components.sumarizattion import load_local_summarizer 
from retrieval.core_retriever import setup_retrievers
from retrieval.retrieval_utils import aplicar_rrf
from indexing.mini_vector_db import gerar_filtro_chroma as gerar_filtro_router
from components.utils_text import format_docs
from generation.llm_response import gerar_resposta_llm
from generation.pipeline_steps import PipelineSteps

# Desliga completamente a telemetria do Chroma
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# ==========================================
# ⚙️ CONFIGURAÇÃO
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
load_dotenv(os.path.join(BASE_DIR, "data", ".env"))

PERSIST_DIRECTORY = "./storage/chroma_db"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "normas_collection")
GLOBAL_STATE = {}

# ==========================================
# 🧩 FUNÇÃO DE EXPANSÃO
# ==========================================
def expandir_contexto(vectorstore: Chroma, docs_rankeados: List[Document], window: int = 1) -> List[Document]:
    """Busca vizinhos para Small-to-Big."""
    docs_expandidos = []
    for doc in docs_rankeados:
        try:
            doc_id = doc.metadata.get("doc_id")
            idx_atual = doc.metadata.get("chunk_index")
            
            if doc_id is None or idx_atual is None:
                docs_expandidos.append(doc)
                continue

            indices_alvo = list(range(int(idx_atual) - window, int(idx_atual) + window + 1))
            
            vizinhos = vectorstore.get(
                where={"$and": [{"doc_id": doc_id}, {"chunk_index": {"$in": indices_alvo}}]},
                include=["documents", "metadatas"]
            )
            
            if not vizinhos['ids']:
                docs_expandidos.append(doc)
                continue
                
            combined = sorted(zip(vizinhos['metadatas'], vizinhos['documents']), key=lambda x: x[0]['chunk_index'])
            texto_completo = "\n".join([texto for _, texto in combined])
            
            novo_doc = Document(page_content=texto_completo, metadata=doc.metadata)
            novo_doc.metadata["expanded"] = True
            docs_expandidos.append(novo_doc)
            
        except Exception:
            docs_expandidos.append(doc)
    return docs_expandidos

# ==========================================
# 📦 INICIALIZAÇÃO
# ==========================================
def carregar_componentes_globais() -> dict:
    if GLOBAL_STATE: return GLOBAL_STATE

    print("[INIT] Carregando componentes...")
    embeddings = get_embedding_function()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    try:
        all_docs_data = vectorstore.get(include=["documents", "metadatas"])
        documentos_bm25 = [Document(page_content=c, metadata=m) for c, m in zip(all_docs_data["documents"], all_docs_data["metadatas"])]
    except Exception as e:
        print(f"❌ [INIT] Erro ao carregar documentos para o BM25: {e}")
        documentos_bm25 = []

    # Configuração do BGE (Cross Encoder)
    try:
        reranker = FlashRankReranker(top_n=5, threshold=0.2) # Top 5 + Corte de Lixo
    except:
        reranker = None

    GLOBAL_STATE.update({
        "vectorstore": vectorstore,
        "documentos_bm25": documentos_bm25, 
        "embeddings_model": embeddings,
        "reranker": reranker
    })
    return GLOBAL_STATE

# ==========================================
# 🔍 RETRIEVAL INSTRUMENTADO
# ==========================================
def _executar_retrieval_avancado(query: str, documentos_filtro: list, etapas: PipelineSteps, indices: dict) -> List[Document]:
    state = GLOBAL_STATE
    
    # 1. Setup Retrievers (Rápido)
    filtro_metadata = gerar_filtro_router(documentos_filtro)
    retriever_semantico, retriever_bm25 = setup_retrievers(state, filtro_metadata)

    # 2. Busca Semântica (Chroma)
    etapas.start_step(indices['semantica'])
    t0 = time.time()
    docs_semanticos = retriever_semantico.invoke(query)
    etapas.complete_step(indices['semantica'], time.time() - t0)

    # 3. Busca Lexical (BM25)
    etapas.start_step(indices['lexical'])
    t0 = time.time()
    docs_textuais = retriever_bm25.invoke(query)
    etapas.complete_step(indices['lexical'], time.time() - t0)

    # Fallback
    if not docs_semanticos and filtro_metadata:
        print("[WARN] Fallback global acionado.")
        retriever_fallback, _ = setup_retrievers(state, filtro_metadata=None)
        docs_semanticos = retriever_fallback.invoke(query)

    # 4. Fusão e Rerank
    etapas.start_step(indices['rerank'])
    t0 = time.time()
    
    docs_fusao = aplicar_rrf([docs_semanticos, docs_textuais])
    
    reranker = state.get("reranker")
    if reranker and docs_fusao:
        candidatos = docs_fusao[:50] 
        docs_finais = reranker.compress_documents(documents=candidatos, query=query)
    else:
        docs_finais = docs_fusao[:5]
        
    etapas.complete_step(indices['rerank'], time.time() - t0)

    return docs_finais

# ==========================================
# 🚀 CONTROLLER PRINCIPAL
# ==========================================
def run_rag_pipeline_hibrido(pergunta_usuario: str) -> dict:
    carregar_componentes_globais()
    
    # Inicializa Steps
    etapas = PipelineSteps()
    
    # Define TODAS as etapas aqui para aparecerem na ordem correta no Frontend
    idx_rewrite = etapas.add_step("Reescrevendo consulta", "Otimização com Gemini")
    idx_semantica = etapas.add_step("Busca Semântica", "Embeddings E5-Base")
    idx_lexical = etapas.add_step("Busca Lexical", "BM25 (Palavras-chave)")
    idx_rerank = etapas.add_step("Fusão & Reranking", "RRF + BGE Cross-Encoder")
    idx_expand = etapas.add_step("Expansão de Contexto", "Small-to-Big (Janela 1)")
    idx_geracao = etapas.add_step("Gerando Resposta", "LLM com contexto")

    try:
        # A. Reescrita
        etapas.start_step(idx_rewrite)
        t0 = time.time()
        pergunta_reescrita, documentos_filtro = preparar_consulta(pergunta_usuario, GLOBAL_STATE["embeddings_model"])
        etapas.complete_step(idx_rewrite, time.time() - t0)

        # B. Retrieval Detalhado
        indices_retrieval = {
            'semantica': idx_semantica,
            'lexical': idx_lexical,
            'rerank': idx_rerank
        }
        
        docs_ranqueados = _executar_retrieval_avancado(
            pergunta_reescrita, 
            documentos_filtro,
            etapas,
            indices_retrieval
        )
        
        # C. Expansão (Small to Big)
        etapas.start_step(idx_expand)
        t0 = time.time()
        if docs_ranqueados:
            docs_finais = expandir_contexto(GLOBAL_STATE["vectorstore"], docs_ranqueados, window=1)
        else:
            docs_finais = []
        etapas.complete_step(idx_expand, time.time() - t0)

        if not docs_finais:
            return {"resposta": "Não encontrei informações relevantes.", "docs_ranqueados": [], "etapas": etapas.get_steps_dict()}

        # D. Geração
        etapas.start_step(idx_geracao)
        t0 = time.time()
        contexto = format_docs(docs_finais)
        resposta = gerar_resposta_llm(pergunta_reescrita, contexto)
        etapas.complete_step(idx_geracao, time.time() - t0)
        
        return {
            "resposta": resposta,
            "contexto_usado": contexto,
            "docs_ranqueados": docs_finais,
            "tempo_total": f"{etapas.summary()['total_duration']:.2f}s",
            "etapas": etapas.get_steps_dict(),
            "pergunta_reescrita": pergunta_reescrita,
            "documentos_filtrados": documentos_filtro
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        return {"resposta": "Erro interno no sistema RAG.", "etapas": etapas.get_steps_dict()}