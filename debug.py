import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

# --- Configuração de Caminhos ---
# Garante que conseguimos importar os módulos da pasta src
# Configuração de Caminhos (Path)
# Adiciona a pasta /src ao PYTHONPATH para importar módulos internos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from components.embeddings import get_embedding_function

# Carrega variáveis de ambiente
ENV_PATH = os.path.join(BASE_DIR, "data", ".env")
load_dotenv(dotenv_path=ENV_PATH)

# Configurações do Banco (Mesmos valores do seu pipeline)
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "storage", "qdrant") 
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "normas_collection")

def inspecionar_metadados():
    print("🔍 Conectando ao ChromaDB...")
    print(f"📂 Diretório: {PERSIST_DIRECTORY}")
    print(f"📦 Coleção: {COLLECTION_NAME}")
    
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            persist_directory=PERSIST_DIRECTORY,
        )
        
        # Pega uma amostra de 5 documentos para não poluir o terminal
        # O Chroma retorna um dicionário com listas
        dados = vectorstore.get(limit=5, include=["metadatas", "documents"])
        
        if not dados['ids']:
            print("\n❌ O banco está VAZIO! Nenhuma indexação foi encontrada.")
            print("   -> Rode o indexer_offline.py primeiro.")
            return

        print(f"\n✅ Conexão bem sucedida! Total de chunks encontrados na amostra: {len(dados['ids'])}\n")
        print("="*60)

        for i, meta in enumerate(dados['metadatas']):
            print(f"--- Chunk {i+1} ---")
            print(f"📂 Arquivo Físico (source): '{meta.get('arquivo', 'N/A')}'")
            
            # ESTE É O CAMPO CRÍTICO PARA O FILTRO
            chave_filtro = meta.get('documento_completo', 'N/A')
            print(f"🏷️  Chave de Filtro (documento_completo): '{chave_filtro}'") 
            
            print(f"📍 Hierarquia: '{meta.get('hierarquia', 'N/A')}'")
            print("-" * 60)

        print("\n💡 ANÁLISE TÉCNICA:")
        print("   O nome que aparece em 'Chave de Filtro' DEVE ser idêntico")
        print("   caractere-por-caractere ao que aparece no log do Router.")
        print("   Se um for 'NOR 007' e o outro 'NOR 007.pdf', o filtro falha.")

    except Exception as e:
        print(f"\n❌ Erro ao ler o banco: {e}")

if __name__ == "__main__":
    inspecionar_metadados()