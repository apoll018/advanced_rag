# Diz para o sistema: "Não tente conectar na internet. Use o que tem no disco."
import os
os.environ['HF_HUB_OFFLINE'] = '1' 
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

"""
Módulo de Embeddings - Serafim 100M (PT-BR)
-------------------------------------------
Utiliza o modelo 'PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir'.
Especialista em Busca Semântica (IR) em Português.
Tamanho: ~100M parâmetros (Equilibrado para CPU).
"""
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
load_dotenv(os.path.join(BASE_DIR, "data", ".env"))

# ============================================================
# CACHE GLOBAL (SINGLETON)
# ============================================================
_MODELO_EMBEDDING_CACHE = None

import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# CLASSE WRAPPER PARA O E5 (AUTOMAGIC PREFIX)
# ============================================================
class E5EmbeddingsWrapper(HuggingFaceEmbeddings):
    """
    Wrapper específico para modelos E5 que exigem prefixos 'query:' e 'passage:'.
    Isso garante a performance assimétrica correta do modelo.
    """
    def embed_query(self, text: str) -> List[float]:
        # Injeta o prefixo de consulta automaticamente
        # O modelo E5 precisa disso para entender que é uma pergunta
        return super().embed_query(f"query: {text}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Injeta o prefixo de documento para indexação
        # O modelo E5 precisa disso para entender que é conteúdo para ser buscado
        processed_texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(processed_texts)

# ============================================================
# CACHE GLOBAL (SINGLETON)
# ============================================================
_MODELO_EMBEDDING_CACHE = None

def get_embedding_function():
    """
    Retorna o modelo de embeddings E5 com tratamento automático de prefixos.
    Usa cache em memória (Singleton).
    """
    global _MODELO_EMBEDDING_CACHE

    if _MODELO_EMBEDDING_CACHE is not None:
        return _MODELO_EMBEDDING_CACHE

    print("⏳ [INIT] Carregando Embeddings E5-Base (Singleton com Wrapper)...")
    
    model_name = r"C:\Users\51-02429\.cache\huggingface\hub\models--intfloat--multilingual-e5-base\snapshots\835193815a3936a24a0ee7dc9e3d48c1fbb19c55"
    
    # Instanciamos a NOSSA classe Wrapper, não a padrão
    _MODELO_EMBEDDING_CACHE = E5EmbeddingsWrapper(
        model_name=model_name,
        model_kwargs={
            "device": "cpu",
            "local_files_only": True
        }, # Mude para 'cuda' se tiver GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print(f"✅ [INIT] Modelo {model_name} carregado e pronto!")
    return _MODELO_EMBEDDING_CACHE