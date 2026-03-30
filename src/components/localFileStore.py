# === localFileStore.py ===
import os
import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_core.stores import InMemoryStore

# para testes depois from langchain_community.storage import InMemoryStore
def salvar_store(store: InMemoryStore, caminho_arquivo: str):
    """
    Salva o conteúdo interno do InMemoryStore (dict) em disco via pickle.
    
    """
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    with open(caminho_arquivo, "wb") as f:
        pickle.dump(store.store, f)
    print(f"Docstore salvo em {caminho_arquivo} ({len(store.store)} registros).")


def carregar_store(caminho_arquivo: str) -> InMemoryStore:
    """
    Carrega um InMemoryStore salvo via pickle.
    Retorna uma instância funcional de InMemoryStore.
    
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo {caminho_arquivo} não encontrado.")
    
    with open(caminho_arquivo, "rb") as f:
        dados = pickle.load(f)

    store = InMemoryStore()
    store.store = dados
    print(f"Docstore carregado de {caminho_arquivo} ({len(store.store)} registros).")
    return store
