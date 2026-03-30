"""
Módulo de Reranking (Refinamento)
---------------------------------
Utiliza o FlashRank (baseado em ONNX/T5) para reordenar os documentos recuperados.
É otimizado para execução em CPU, sendo muito mais rápido que Cross-Encoders tradicionais.
"""

"""
Módulo de Reranking (Refinamento)
---------------------------------
Utiliza o FlashRank (baseado em ONNX/T5) para reordenar documentos recuperados.
É otimizado para execução em CPU, sendo muito mais rápido que Cross-Encoders tradicionais.
"""
import os
import logging
from typing import List
from pathlib import Path

# 1. Configurações de Ambiente (Sempre no topo, antes dos imports pesados)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Resolve o caminho do cache de forma dinâmica (funciona em qualquer PC/Servidor)
# Exemplo: pega a pasta do usuário atual e adiciona .cache/huggingface
USER_HOME = Path.home()
HF_CACHE_DIR = USER_HOME / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

# 2. Imports das bibliotecas pesadas
from transformers import logging as hf_logging
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Silencia os avisos do HuggingFace
hf_logging.set_verbosity_error()

class NeuralReranker:
    """
    Reranker Neural baseado em Cross-Encoder (Sentence-Transformers).
    Usa o BGE-M3 (Estado da Arte).
    """
    def __init__(self, target_model: str = None, top_n: int = 5, threshold: float = 0.2):
        # Caminho dinâmico para o modelo local
        if target_model is None:
            self.target_model = str(HF_CACHE_DIR / "hub" / "models--BAAI--bge-reranker-v2-m3" / "snapshots" / "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
        else:
            self.target_model = target_model
            
        self.top_n = top_n
        self.threshold = threshold
        self.model = None
        
        print(f"⏳ [RERANK] Carregando Cross-Encoder: {self.target_model}...")
        
        try:
            # Inicializa apenas UMA vez, dentro do bloco de proteção
            # Se for rodar em GPU, basta trocar 'cpu' por 'cuda'
            self.model = CrossEncoder(
                self.target_model, 
                max_length=512, 
                device='cpu',
                local_files_only=True
            )
            print(f"✅ [RERANK] Modelo BGE carregado! (Top-N={top_n}, Min-Score={threshold})")
        except Exception as e:
            print(f"❌ [RERANK] Erro ao carregar modelo local: {e}")
            # Você pode adicionar um fallback aqui se quiser baixar da internet caso falhe

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Reordena e FILTRA os documentos baseando-se na relevância semântica.
        """
        # Proteção caso o modelo não tenha carregado ou a lista esteja vazia
        if not self.model or not documents:
            return documents[:self.top_n]

        # 1. Predição dos Scores
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        # 2. Associação e Filtragem
        docs_with_scores = []
        for doc, score in zip(documents, scores):
            f_score = float(score)
            
            # Salva o score no metadado para debug e rastreabilidade
            doc.metadata["score_rerank"] = f_score
            
            # Só aceita se o score for maior ou igual ao limiar estabelecido
            if f_score >= self.threshold:
                docs_with_scores.append((doc, f_score))

        # 3. Ordenação (Maior score primeiro)
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Corte Top-N
        top_docs = [doc for doc, _score in docs_with_scores[:self.top_n]]

        # Log de debug opcional
        if len(documents) > len(top_docs):
            _removidos = len(documents) - len(top_docs)
            # print(f"ℹ️ [RERANK] {_removidos} documentos ignorados (abaixo do threshold ou fora do top-{self.top_n}).")

        return top_docs

# Mantém compatibilidade com implementações antigas, se necessário
class FlashRankReranker(NeuralReranker):
    pass

""" class FlashRankReranker:
    
    AVAILABLE_MODELS = [
        "rank-T5-flan",
        "ms-marco-MiniLM-L-12-v2"
    ]

    def __init__(
        self, 
        model_name: str = "rank-T5-flan", 
        top_n: int = 5,
        cache_dir: str = "./storage/flashrank_models"
    ):

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' não suportado. "
                f"Opções: {', '.join(self.AVAILABLE_MODELS)}"
            )

        logger.info(f"[INIT] Carregando Reranker FlashRank ({model_name})...")
        try:
            self.ranker = Ranker(
                model_name=model_name,
                cache_dir=cache_dir
            )
            self.model_name = model_name
            self.top_n = top_n
            logger.info(f"✅ FlashRank pronto. Modelo={model_name}, Top-N={top_n}")
        except Exception as e:
            logger.error(f"❌ Falha ao carregar FlashRank: {e}")
            self.ranker = None

    def compress_documents(
        self, 
        documents: List[Document], 
        query: str
    ) -> List[Document]:

        if not documents:
            logger.debug("compress_documents chamado com lista vazia")
            return []
        
        if not query or not query.strip():
            raise ValueError("Query não pode estar vazia")
        
        if self.ranker is None:
            logger.warning("Reranker não disponível. Retornando docs originais com fallback marker.")
            for doc in documents[:self.top_n]:
                doc.metadata['score_rerank'] = -1
                doc.metadata['rerank_status'] = 'disabled'
            return documents[:self.top_n]

        start = datetime.now()
        logger.debug(f"Reranking {len(documents)} docs para: {query[:60]}...")

        # Formata para o padrão que o FlashRank espera
        passages = [
            {
                "id": idx,
                "text": doc.page_content,
                "meta": doc.metadata  # Preserva metadados originais
            }
            for idx, doc in enumerate(documents)
        ]

        try:
            request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(request)
        except Exception as e:
            logger.error(f"❌ FlashRank falhou: {e}")
            # Fallback: retornar docs originais com marcador de erro
            for doc in documents[:self.top_n]:
                doc.metadata['score_rerank'] = -1
                doc.metadata['rerank_status'] = 'error'
            return documents[:self.top_n]

        # Reconstrói Documents com scores
        final_docs = []
        for res in results[:self.top_n]:
            metadata = documents[res["id"]].metadata.copy()
            metadata["score_rerank"] = float(res["score"])

            doc = Document(
                page_content=documents[res["id"]].page_content,
                metadata=metadata
            )
            doc.metadata['rerank_status'] = 'success'

            final_docs.append(doc)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(
            f"✅ Rerank completo: {len(documents)} → {len(final_docs)} "
            f"em {elapsed:.2f}s (modelo={self.model_name})"
        )

        return final_docs """

"""
import os
import sys
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

# Importamos PrivateAttr e Field
from pydantic import ConfigDict, Field, PrivateAttr
from sentence_transformers import CrossEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class CustomCrossEncoderReranker(BaseDocumentCompressor):
    model_name: str = Field(default="Alibaba-NLP/gte-multilingual-reranker-base")
    top_n: int = Field(default=5)

    _model: CrossEncoder = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = CrossEncoder(self.model_name, trust_remote_code=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:

        if not documents:
            return []

        # ✅ CORRIGIR: Converter generator para lista
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)  # Passa lista, não generator

        # Atribui scores e reordena
        for doc, score in zip(documents, scores):
            doc.metadata["crossencoder_score"] = float(score)

        return sorted(
            documents,
            key=lambda d: d.metadata["crossencoder_score"],
            reverse=True,
        )[: self.top_n] """


""" class CustomCrossEncoderReranker(BaseDocumentCompressor):
    # Campos Pydantic que DEVEM ser inicializados na chamada do construtor
    model_name: str = Field(default="Alibaba-NLP/gte-multilingual-reranker-base")
    top_n: int = Field(default=5)
    
    _model: CrossEncoder = PrivateAttr()
    
    # Permite o CrossEncoder ser um tipo arbitrário
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        # 1. Pydantic inicializa apenas os campos públicos (model_name, top_n)
        super().__init__(**kwargs) 
        
        # 2. Inicializa o modelo real no atributo privado (_model)
        self._model = CrossEncoder(self.model_name, trust_remote_code=True)
        
    def compress_documents(
        self, 
        documents: Sequence[Document], 
        query: str, 
        callbacks: Callbacks | None = None, 
    ) -> Sequence[Document]:
        
        # rdena os documentos por relevância, mantendo apenas os top_n.
        
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs) 

        for i, doc in enumerate(documents):
            doc.metadata["crossencoder_score"] = float(scores[i])

        reranked_docs = sorted(documents, key=lambda d: d.metadata["crossencoder_score"], reverse=True)
        return reranked_docs[: self.top_n] """