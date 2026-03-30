"""
Project: Advanced RAG Pipeline
Author: Lucas Henrique da S. Medeiros
GitHub: https://github.com/apoll018
Created: 2026

Este código é de autoria original. Uso sujeito à licença do projeto.
"""

import os
import sys
import uuid
import hashlib
import pickle
from typing import Dict, Any, List

import numpy as np
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

collection_name = os.getenv("CHROMA_COLLECTION", "normas_collection")
CHUNK_CONTENT_LIMIT = 500  # Limite de caracteres para preview

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

# Configurações
ENV_PATH = os.path.join(BASE_DIR, "data", ".env")
load_dotenv(dotenv_path=ENV_PATH)

CAMINHO_DB = os.getenv("MINI_VECT_DB_PATH", os.path.join(BASE_DIR, "data", "mini_vector_db.pkl"))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "storage", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "normas_collection")
CHUNK_CONTENT_LIMIT = 400 

# Imports locais
try:
    from components.embeddings import get_embedding_function
    from langchain_chroma import Chroma
    from chromadb import PersistentClient
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    sys.exit(1)

# ==============================================================
# === 1. INSPEÇÃO DO VECTORSTORE (CHROMA) ======================
# ==============================================================
def inspecionar_vectorstore(persist_dir: str, collection_name: str) -> None:
    print("\n" + "=" * 90)
    print(f"🔎 INSPECIONANDO VECTORSTORE (Small-to-Big) → {persist_dir}")
    print("=" * 90)

    try:
        # Usamos o client nativo do Chroma para estatísticas rápidas
        client = PersistentClient(path=persist_dir)
        try:
            coll = client.get_collection(collection_name)
        except ValueError:
            print(f"❌ Coleção '{collection_name}' não encontrada!")
            return

        count = coll.count()
        print(f"📚 Total de Chunks no Banco: {count}")

        # --- CORREÇÃO AQUI: Removemos "ids" do include ---
        # O Chroma retorna IDs automaticamente, não precisamos pedir.
        results = coll.get(limit=5, include=["metadatas", "documents"])
        
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        docs = results.get("documents", [])

        if not ids:
            print("⚠️ O banco está vazio.")
            return

        for i in range(len(ids)):
            meta = metadatas[i]
            content = docs[i]
            size = len(content)
            
            # Preview do texto
            preview = content[:CHUNK_CONTENT_LIMIT]
            if size > CHUNK_CONTENT_LIMIT: preview += "..."

            # Metadados da Nova Arquitetura
            doc_group_id = meta.get('doc_id', '❌ N/A')
            chunk_idx = meta.get('chunk_index', '❌ N/A')
            total_chunks = meta.get('total_chunks', '?')
            hierarquia = meta.get('hierarquia', 'Geral')
            arquivo = meta.get('arquivo', 'Desconhecido')

            print("\n" + "-" * 70)
            print(f"🔹 CHUNK #{i+1} | ID Chroma: {ids[i][:8]}...")
            print("-" * 70)
            
            print(f"📂 Arquivo:     {arquivo}")
            print(f"🔗 Grupo (DocID): {doc_group_id}")
            print(f"🔢 Sequência:   {chunk_idx} de {total_chunks}")
            print(f"📌 Hierarquia:  {hierarquia}")
            
            print(f"\n📝 Tamanho: {size} chars")
            print("📄 Conteúdo (Preview):")
            print(f"   {preview}")

        # --- Verificação de Arquivos Únicos ---
        print("\n📂 Resumo de Arquivos Indexados:")
        # Aqui precisamos pegar apenas metadados, sem docs, para ser rápido
        all_data = coll.get(include=["metadatas"])
        all_meta = all_data["metadatas"]
        
        arquivos_unicos = {}
        for m in all_meta:
            nome = m.get("arquivo", "Desconhecido")
            if nome not in arquivos_unicos:
                arquivos_unicos[nome] = 0
            arquivos_unicos[nome] += 1
            
        for arq, qtd in sorted(arquivos_unicos.items()):
            print(f"   - {arq} ({qtd} chunks)")

    except Exception as e:
        print(f"❌ Erro ao inspecionar VectorStore: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================
# === 2. INSPEÇÃO DO MINI-DB (ROUTER) ==========================
# ==============================================================
def inspecionar_mini_db(caminho: str, max_docs: int = 5):
    print("\n" + "="*90)
    print(f"🔍 INSPEÇÃO DO MINI VECTOR DB (Router) → {caminho}")
    print("="*90)

    if not os.path.exists(caminho):
        print("❌ Arquivo não encontrado!")
        return

    try:
        with open(caminho, "rb") as f:
            mini_db = pickle.load(f)
    except Exception as e:
        print(f"❌ Erro ao ler pickle: {e}")
        return

    print(f"📂 Total de Documentos Roteáveis: {len(mini_db)}")
    
    for i, (doc_name, termos) in enumerate(mini_db.items()):
        if i >= max_docs: break
        
        termos_txt = [t['term'] for t in termos[:5]]
        print(f"\n📄 {doc_name}")
        print(f"   • Termos ({len(termos)}): {termos_txt} ...")
        
        # Valida shape do primeiro vetor
        if termos and 'vec' in termos[0]:
            shape = termos[0]['vec'].shape
            print(f"   • Vector Shape: {shape} (OK)" if shape else "   • Vector Shape: ❌ Erro")

# ==============================================================
# === 3. TESTE DE EMBEDDINGS (CACHE) ===========================
# ==============================================================
def testar_embeddings_cache():
    print("\n" + "="*90)
    print("⚡ TESTE DE EMBEDDINGS & CACHE")
    print("="*90)
    
    print("1️⃣ Primeira chamada (deve carregar modelo)...")
    emb = get_embedding_function()
    
    print("2️⃣ Segunda chamada (deve ser instantânea)...")
    emb2 = get_embedding_function()
    
    if emb is emb2:
        print("✅ Singleton funcionando! (Mesma instância de memória)")
    else:
        print("⚠️ Singleton falhou (Instâncias diferentes).")

    # Teste de Sanidade
    vec = emb.embed_query("Teste de sanidade")
    print(f"✅ Vetor gerado com sucesso. Dimensão: {len(vec)}")

# ==============================================================
# === MAIN =====================================================
# ==============================================================
if __name__ == "__main__":
    # 1. Teste de Hardware/Modelo
    testar_embeddings_cache()
    
    # 2. Vectorstore (Onde ficam os dados reais)
    inspecionar_vectorstore(PERSIST_DIRECTORY, COLLECTION_NAME)
    
    # 3. Router (Onde ficam as tags)
    inspecionar_mini_db(CAMINHO_DB)
    
    print("\n✅ Auditoria finalizada.\n")