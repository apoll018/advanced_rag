import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import uuid
from datetime import datetime
from typing import Dict, Any, List
# Nota: LangChain recomenda usar as classes 'core' e 'community' no lugar de 'classic'
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
# Importa componentes internos
# Nota: GLOBAL_STATE e carregar_componentes_globais devem ser definidas em rag_pipeline.py
from app.rag_pipeline import GLOBAL_STATE, carregar_componentes_globais 
from components.localFileStore import salvar_store

# ==============================================================
# 🔄 FUNÇÃO: INDEXAR UM ÚNICO DOCUMENTO (para Streamlit)
# ==============================================================
def indexar_novo_documento(file_path: str, metadados: Dict[str, Any]) -> bool:
    """
    Indexa um único documento PDF utilizando o padrão Small-to-Big Chunking
    (Parent Document Retriever) no Chroma e no DocStore. 
    
    Acessa os componentes de estado global (vectorstore, docstore, splitters).

    Args:
        file_path (str): Caminho para o arquivo PDF a ser indexado.
        metadados (Dict[str, Any]): Metadados personalizados fornecidos via Streamlit.

    Returns:
        bool: True se a indexação foi bem-sucedida, False caso contrário.
    """
    filename = os.path.basename(file_path)

    # 1️⃣ Carregar o PDF usando PyPDFLoader
    print(f"Iniciando carregamento do arquivo: {filename}")
    try:
        loader = PyPDFLoader(file_path)
        docs: List[Document] = loader.load() # Retorna uma lista de documentos (um por página)
        print(f"Documento carregado. Total de {len(docs)} páginas.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo PDF {filename}: {e}")
        return False

    # 2️⃣ Adicionar Metadados (Automáticos e Personalizados)
    doc_id_pai = str(uuid.uuid4()) # ID único para agrupar todos os chunks deste documento
    try:
        tamanho_kb = os.path.getsize(file_path) // 1024
    except Exception:
        tamanho_kb = -1 # Trata erro caso o arquivo já tenha sido movido/excluído

    for doc in docs:
        # Metadados automáticos (gerados pelo sistema)
        doc.metadata.update({
            "arquivo": filename,
            "doc_id": doc_id_pai, # Usado para referenciar o documento pai
            "data_upload": datetime.now().isoformat(), # Data/hora da indexação
            "num_paginas": len(docs),
            "tamanho_kb": tamanho_kb
        })
        # Metadados personalizados (vindos do formulário do Streamlit)
        doc.metadata.update(metadados)
        
        # Pré-processamento de Conteúdo para E5 (Embedding Model Optimization)
        # Adiciona o prefixo 'passage: ' para otimizar o desempenho do modelo E5
        doc.page_content = "passage: " + doc.page_content

    # 3️⃣ Garantir que o Estado Global está Inicializado
    if not GLOBAL_STATE:
        print("Componentes globais (VectorStore, DocStore) não carregados. Inicializando...")
        carregar_componentes_globais()

    # Extrai componentes do estado global
    vectorstore = GLOBAL_STATE.get("vectorstore")
    docstore = GLOBAL_STATE.get("docstore")
    pai_splitter = GLOBAL_STATE.get("pai_splitter")
    filho_splitter = GLOBAL_STATE.get("filho_splitter")

    if not vectorstore or not docstore or not pai_splitter or not filho_splitter:
        print("Erro: Componentes essenciais (Chroma, DocStore ou Splitters) não foram carregados corretamente.")
        return False

    # 4️⃣ Criar ParentDocumentRetriever
    # Configura o retriever responsável por indexar: chunks menores (filhos) no vetor, 
    # e chunks maiores (pais) no DocStore.
    retriever_indexador = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=pai_splitter,
        child_splitter=filho_splitter,
    )

    # 5️⃣ Adicionar os documentos (pages/chunks) no banco de dados
    try:
        print(f"Iniciando adição de documentos ao DocStore e VectorStore (ParentDocumentRetriever)...")
        retriever_indexador.add_documents(docs)
        #vectorstore.persist()
    except Exception as e:
        print(f"Erro ao indexar documentos: {e}")
        return False

    # 6️⃣ Atualizar a Lista de Documentos para o BM25 (Retrieval Híbrido)
    # É necessário reconstruir ou atualizar o conjunto de documentos
    # que o retriever lexical (BM25) utiliza, garantindo que o novo conteúdo esteja presente.
    try:
        print("Iniciando atualização dos documentos para o retriever BM25...")
        # Obtém todos os documentos (chunks filhos) do VectorStore
        docs_all = vectorstore.get(include=["metadatas", "documents"]) 
        documentos_bm25 = [
            Document(page_content=c, metadata=m)
            for c, m in zip(docs_all["documents"], docs_all["metadatas"])
        ]
        # Atualiza o estado global com a nova lista completa
        GLOBAL_STATE["documentos_bm25"] = documentos_bm25 
        print(f"BM25: {len(documentos_bm25)} chunks disponíveis para busca lexical.")
    except Exception as e:
        print(f"Aviso: Não foi possível atualizar o BM25: {e}")

    print(f"Documento '{metadados.get('documento_completo', filename)}' indexado com sucesso!")
    
    # 7️⃣ Persistir o DocStore (para manter o mapeamento Parent-Child)
    try:
        caminho_docstore = os.path.join(os.getenv("DOCSTORE_PATH", "./storage/storage"), "docstore.pkl")
        salvar_store(docstore, caminho_docstore)
        print("DocStore salvo com sucesso no disco.")
    except Exception as e:
        print(f"Erro ao salvar DocStore: {e}")
        
    return True