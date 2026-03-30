"""
Módulo Indexador Offline (Ingestão de Documentos)
-------------------------------------------------
Pipeline ETL ajustado para Small-to-Big Local (Context Expansion).
Elimina ParentDocumentRetriever e DocStore separado.
"""

import os
import sys
import shutil
import hashlib
import uuid
import re
from datetime import datetime
from typing import List, Any

from dotenv import load_dotenv
from markitdown import MarkItDown
from pdf2docx import Converter

# --- MUDANÇA: Usamos apenas Chroma e Document ---
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import unicodedata
from chromadb.config import Settings

# --- Imports Internos ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

from components.embeddings import get_embedding_function
# from components.localFileStore import carregar_store, salvar_store  <-- REMOVIDO
# from components.splitters import get_child_splitter, get_parent_splitter <-- REMOVIDO (Usamos lógica local)
from indexing.mini_vector_db import atualizar_mini_db
from components.splitters import get_main_splitter, CHUNK_SIZE

# ==========================================
# ⚙️ CONFIGURAÇÕES
# ==========================================
load_dotenv(os.path.join(BASE_DIR, "data", ".env"))

PERSIST_DIRECTORY = os.path.join(BASE_DIR, "storage", "chroma_db")
# DOCSTORE_DIRECTORY removido pois não é mais necessário
MINI_DB_PATH = os.path.join(BASE_DIR, "data", "mini_vector_db.pkl")
PDFS_DIR = os.getenv("PDFS_PATH", "./data/pdfs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "normas_collection")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ==========================================
# 📝 REPORTING
# ==========================================
class IndexingReporter:
    def __init__(self, filename_prefix: str):
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(LOGS_DIR, f"report_{filename_prefix}_{timestamp}.md")
        self.buffer = []
        self._header()
        
    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    def _header(self):
        self.buffer.append(f"# Relatório de Indexação\n")
        self.buffer.append(f"**Gerado em:** {datetime.now()}\n")
        self.buffer.append(f"**Arquivo de log:** `{self.filepath}`\n")
        self.buffer.append("---\n")
        
    # ---------------------------------------------------------
    # CAMPOS SIMPLES
    # ---------------------------------------------------------

    def log_kv(self, key: str, value: str):
        self.buffer.append(f"\n## {key}\n\n{value}\n")

    def log_code_block(self, title: str, content: str, lang: str = "markdown"):
        self.buffer.append(f"\n## {title}\n\n```{lang}\n{content}\n```\n")
        
    # ---------------------------------------------------------
    # AMOSTRA DE CHUNKS — MOSTRA COMPLETO, FORMATA BONITO
    # ---------------------------------------------------------

    def log_chunks_sample(self, chunks: List[Document], n_samples: int = 5):
        total = len(chunks)
        n = min(n_samples, total)
        self.buffer.append(f"\n## Amostra de Chunks ({n} de {total})\n")

        for i, doc in enumerate(chunks[:n]):
            meta = doc.metadata or {}
            idx = meta.get("chunk_index", "N/A")
            doc_id = meta.get("doc_id", "N/A")
            tamanho = len(doc.page_content)

            self.buffer.append(
                f"\n### Chunk {i+1} (Index: {idx})\n"
                f"- **Doc ID (Group):** `{doc_id}`\n"
                f"- **Tamanho:** {tamanho} caracteres\n"
                f"- **Metadados:** `{meta}`\n"
            )
            linhas = doc.page_content.split("\n")
            preview = "\n".join(linhas[:10]) 
            self.buffer.append(f"\n```markdown\n{preview}\n```\n")

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.buffer))
        print(f"[LOG] Relatório salvo: {os.path.basename(self.filepath)}")

# ==========================================
# 🛠️ UTILITÁRIOS
# ==========================================
def get_file_hash(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

def limpar_banco_dados():
    # Removemos DOCSTORE_DIRECTORY da lista de limpeza
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f"[RESET] Apagado: {PERSIST_DIRECTORY}")
    if os.path.exists(MINI_DB_PATH):
        os.remove(MINI_DB_PATH)
        print(f"[RESET] Apagado: {MINI_DB_PATH}")

def remover_chunks_antigos(vectorstore: Chroma, arquivo: str):
    """
    Remove documentos antigos baseados no metadado 'arquivo'.
    Agora opera apenas no Vectorstore.
    """
    print(f"   [CLEAN] Verificando versão antiga...")
    try:
        # Busca IDs onde metadado 'arquivo' == arquivo atual
        results = vectorstore.get(where={"arquivo": arquivo}, include=["metadatas"])
        if not results["ids"]: 
            return

        ids_to_delete = results["ids"]
        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
            print(f"   [CLEAN] Removidos {len(ids_to_delete)} chunks antigos.")
            
    except Exception as e:
        print(f"   [ERROR] Erro na limpeza: {e}")

# ... (Funções de extração de nome e regex mantidas iguais) ...
def extrair_codigo_documento(nome_arquivo: str) -> str:
    
    """
    Extrai o código do documento a partir do nome do arquivo.
    Ex.: 'NOR 032 - Documento.pdf' → 'NOR 032'
    """
    
    padrao = r'\b(NOR|POL|PRO|MANUAL|REG)[\s\-_]*([0-9]{1,3})\b'
    
    match = re.search(padrao, nome_arquivo, re.IGNORECASE)
    
    if match:
        prefixo = match.group(1).upper()
        numero = match.group(2).zfill(3)
        return f"{prefixo} {numero}"
    
    fallback = os.path.splitext(nome_arquivo)[0]
    
    return fallback.upper().strip()

def extrair_tipo_numero(codigo: str):
    
    """
    Recebe algo como 'PRO 003' e retorna ('PRO', '003')
    """
    
    match = re.match(r'(NOR|POL|PRO|MANUAL|REG)\s*(\d+)', codigo, re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2).zfill(3)
    return None, None

def normalizar_nome_pdf(nome: str) -> str:
    
    """
    Normaliza nomes de PDFs para manter um padrão consistente.
    
    Exemplo:
    'PRO 005 – Procedimento...' → 'PRO 005 - Procedimento...'
    'NOR - 011  Norma...' → 'NOR 011 - Norma...'
    """
    
    # 1) Normaliza Unicode (remove variações de travessão)
    nome = unicodedata.normalize("NFKC", nome)

    # 2) Converte travessões e sublinhados para hífen simples
    nome = nome.replace("–", "-").replace("—", "-").replace("_", "-")

    # 3) Remove espaços duplicados
    nome = re.sub(r"\s+", " ", nome)

    # 4) Extrai código caso haja bagunça no começo
    m = re.search(r'\b(NOR|POL|PRO|MANUAL|REG)\s*[- ]*\s*(\d{1,3})\b', nome, re.IGNORECASE)
    
    if m:
        prefixo = m.group(1).upper()
        numero = m.group(2).zfill(3)
        codigo = f"{prefixo} {numero}"
    else:
        base = nome.replace(".pdf", "").strip()
        return base + ".pdf"
    
    # 5) Extrai o restante do nome/título após o código
    resto = nome[m.end():].strip(" -")

    # 6) Monta nome final padronizado
    nome_final = f"{codigo} - {resto}"
    
    # 7) Garante extensão .pdf no final
    if not nome_final.lower().endswith(".pdf"):
        nome_final = nome_final + ".pdf"
    
    # 8) Retorna o nome final padronizado
    return nome_final

# ==========================================
# 🧹 LIMPEZA DE TEXTO
# ==========================================

def remover_cabecalhos_administrativos(text: str) -> str:
    
    # Lista de termos que denunciam um cabeçalho de tabela de norma
    chaves = [r"\|\s*\*\*Norma\*\*", 
              r"\|\s*\*\*Código\*\*", 
              r"\|\s*\*\*Área\*\*", 
              r"\|\s*\*\*Responsável\*\*", 
              r"\|\s*\*\*Vigência.*?\*\*", 
              r"\|\s*\*\*Classificação\*\*", 
              r"\|\s*Norma de Doações"]
    
    # Deleta a linha inteira se encontrar qualquer chave
    pattern = re.compile(r'^.*(?:' + '|'.join(chaves) + r').*$', re.MULTILINE | re.IGNORECASE)
    return re.sub(pattern, '', text)

def limpar_bolinhas_e_lixo(text: str) -> str:
    
    """
    Remove sequências de bolinhas/pontos que poluem o texto,
    sem quebrar listas reais.
    """
    
    # 1. Remove linhas inúteis, mas preserva bullets reais
    pattern_linha_lixo = re.compile(r'^[ \t\|\.\-_]*$', re.MULTILINE)
    text = re.sub(pattern_linha_lixo, '', text)
    
    # 2. Remove sequências artificiais (4+ bolinhas), sem afetar listas
    pattern_sequencia = re.compile(r'(?:[•\.\-_]\s*){4,}')
    return re.sub(pattern_sequencia, ' ', text)

def dissolver_tabelas_falsas(text: str) -> str:
    
    lines = text.split('\n')
    new_lines = []
    
    # Detecta: Romanos, Numéricos, Bullets dentro de tabelas falsas e as dissolve, mantendo o conteúdo
    pattern = re.compile(r'^\|\s*(\([ivxlcdm]+\)|\d+(\.\d+)*\.?|\*\*\d+.*?|[•\-])\s*\|?', re.IGNORECASE)
    
    skip = False
    for line in lines:
        s = line.strip()
        
        if skip:
            if re.match(r'^\|\s*[-:]+\s*\|', s): skip = False; continue
            skip = False
            
        # Remove tabela vazia/inútil de 2 colunas
        if s.count("|") == 2 and s.startswith("|") and s.endswith("|"):
            content = s.strip("|").strip()
            if not content: continue
            new_lines.append(content)
            continue
        
        if pattern.match(s):
            c = line.replace('|', ' ').strip()
            c = re.sub(r'\s{2,}', ' ', c)
            if '**' in c:
                c = re.sub(r'^\s*\*\*(\d+(\.\d+)*\.?)\*\*\s*', r'## \1 ', c).replace('**', '')
            new_lines.append(c)
            skip = True
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def limpar_output_markitdown(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = remover_cabecalhos_administrativos(text)
    text = dissolver_tabelas_falsas(text)
    text = limpar_bolinhas_e_lixo(text)
    text = re.sub(r'^\s*[|: -]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\([ivxlcdm]+\).*?;)\s*(?=\(\s*[ivxlcdm]+\s*\))', r'\1\n', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\*\*(\d+)\s+(\d+)\.?\*\*\s*', r'### \1.\2. ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\*(\d+)\.?\*\*\s*(.*)', r'## \1. \2', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\*(\d+(\.\d+)+)\.?\*\*\s*(.*)', r'### \1 \3', text, flags=re.MULTILINE)
    text = re.sub(r'^(#+\s+.*?)\*\*(.*?)\*\*', r'\1\2', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(Página\s*)?\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ==========================================
# 📄 PARSER
# ==========================================

def converter_pdf_markdown(pdf_path: str, reporter: IndexingReporter) -> str:
    # Mantém a mesma lógica de conversão MarkItDown
    temp_dir = "temp_conversion_process"
    os.makedirs(temp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    docx_path = os.path.join(temp_dir, f"{base_name}.docx")
    
    if not os.path.exists(docx_path):
        try:
            cv = Converter(pdf_path)
            cv.convert(docx_path, start=0, end=None, parse_header_footer=True)
            cv.close()
        except Exception as e:
            reporter.log_kv("Erro Conversão", str(e))
            return ""

    try:
        md = MarkItDown()
        result = md.convert(docx_path)
        raw = result.text_content
    except Exception as e:
        reporter.log_kv("Erro MarkItDown", str(e))
        return ""

    clean = limpar_output_markitdown(raw)
    reporter.log_code_block("Amostra Markdown", clean[:1500] + "\n...")
    
    try: shutil.rmtree(temp_dir)
    except: pass
    
    return clean

def parse_documento(pdf_path: str, meta_base: dict, reporter: IndexingReporter, doc_group_id: str) -> List[Document]:
    md_text = converter_pdf_markdown(pdf_path, reporter)
    if not md_text: return []

    # 1. Split Semântico (Respeita a estrutura do documento)
    # Isso garante que a gente saiba onde começa/termina cada seção
    splitter_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False
    )
    docs_md = splitter_md.split_text(md_text)
    
    # 2. Split Recursivo (Garante o tamanho SMALL para a busca)
    # Aqui usamos o SEU splitter configurado (500 chars)
    rec_splitter = get_main_splitter()
    
    final_docs = []
    
    for d in docs_md:
        # Se o pedaço do markdown já for maior que nosso alvo (ex: 500), a gente quebra.
        # Adicionamos uma margem de segurança (ex: 1.2x) para não quebrar frases à toa se passar só um pouquinho.
        if len(d.page_content) > (CHUNK_SIZE * 1.2): 
            chunks_rec = rec_splitter.split_text(d.page_content)
            # Transforma strings em Documentos mantendo os metadados do pai (cabeçalhos)
            final_docs.extend([
                Document(page_content=t, metadata=d.metadata.copy()) 
                for t in chunks_rec
            ])
        else:
            final_docs.append(d)

    processed = []
    total_chunks = len(final_docs)

    # 3. Indexação Sequencial (Para o Small-to-Big funcionar)
    for i, d in enumerate(final_docs):
        # Limpeza extra de metadados
        h1 = d.metadata.get("h1", "").replace("#", "").strip()
        h2 = d.metadata.get("h2", "").replace("#", "").strip()
        h3 = d.metadata.get("h3", "").replace("#", "").strip()
        hierarquia = " > ".join(filter(None, [h1, h2, h3])) or "Geral"
        
        # Ignora chunks muito pequenos (lixo de conversão)
        if len(d.page_content) < 50: continue 

        new_meta = meta_base.copy()
        new_meta.update({
            "doc_id": doc_group_id,      # ID do Grupo (Arquivo)
            "chunk_index": i,            # ID Sequencial (0, 1, 2...)
            "total_chunks": total_chunks,
            "hierarquia": hierarquia,
            "content_length": len(d.page_content)
        })
        d.metadata = new_meta
        processed.append(d)

    return processed
# ==========================================
# 🚀 ORQUESTRADOR
# ==========================================
def indexar_documentos(reindexar: bool = False, metadados_padrao: dict = None):
    print("\n" + "="*60)
    print("🚀 RAG INDEXER V18 (SMALL-TO-BIG LOCAL)")
    print("="*60)

    if reindexar:
        limpar_banco_dados()

    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # Inicializa Vectorstore diretamente
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=PERSIST_DIRECTORY,
        client_settings=Settings(anonymized_telemetry=False)
    )

    if not os.path.exists(PDFS_DIR):
        print(f"[ERROR] Pasta não encontrada: {PDFS_DIR}")
        return

    pdfs = [f for f in os.listdir(PDFS_DIR) if f.lower().endswith(".pdf")]
    print(f"[INFO] Fila: {len(pdfs)} arquivos.")
    
    for pdf in pdfs:
        nome_normalizado = normalizar_nome_pdf(pdf)
        pdf_path = os.path.join(PDFS_DIR, pdf)
        file_hash = get_file_hash(pdf_path)
        pdf_exibicao = nome_normalizado
        
        codigo = extrair_codigo_documento(pdf)
        nome_canonico = codigo.replace(" ", "_")
        tipo, numero = extrair_tipo_numero(codigo)
        safe_name = os.path.splitext(pdf)[0]

        print("\n📄 Documento encontrado:")
        print(f"Nome original: {pdf_exibicao}")
        print(f"Código:        {codigo}")

        reporter = IndexingReporter(safe_name)
        reporter.log_kv("Arquivo", pdf)
        reporter.log_kv("ID Sistema", nome_canonico)

        # Verifica se já existe pelo hash
        existing = vectorstore.get(where={"arquivo": pdf}, include=["metadatas"])
        if existing['metadatas']:
            old_hash = existing['metadatas'][0].get('arquivo_hash')
            if old_hash == file_hash:
                print(f"   [SKIP] {pdf} (Hash inalterado)")
                reporter.log_kv("Status", "Skipped")
                reporter.save()
                continue
            else:
                remover_chunks_antigos(vectorstore, pdf)

        # Coleta metadados extras se solicitado
        if metadados_padrao is not None:
            print("📝 Informe os metadados para este arquivo:")
            tipo_input = input(f"   Tipo [{tipo}]: ").strip() or tipo
            num_input  = input(f"   Número [{numero}]: ").strip() or numero
            tags_input = input("   Tags: ").strip()

            meta_extra = {}
            if tipo_input: meta_extra["tipo"] = tipo_input
            if num_input: meta_extra["numero"] = num_input
            if tags_input: meta_extra["tags"] = tags_input

            metadados_pdf = {**metadados_padrao, **meta_extra}
        else:
            metadados_pdf = metadados_padrao or {}

        # Gera ID Único para o documento inteiro (CRUCIAL para a estratégia)
        doc_group_id = str(uuid.uuid4())

        meta = {
            "arquivo": pdf_exibicao,
            "arquivo_hash": file_hash,
            "data_upload": datetime.now().isoformat(),
            "documento_completo": nome_canonico,
            **metadados_pdf
        }

        if isinstance(meta.get("tags"), list):
            meta["tags"] = ", ".join(meta["tags"])

        # Chama parser passando o ID de grupo
        chunks = parse_documento(pdf_path, meta, reporter, doc_group_id)
        
        if not chunks:
            print("   [FAIL] Falha na extração.")
            continue

        # Adiciona diretamente ao Chroma (sem DocStore)
        vectorstore.add_documents(chunks)

        # Atualiza Mini-DB para o Router
        tags_raw = meta.get("tags", "")
        if tags_raw:
            tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
            atualizar_mini_db(nome_canonico, tags_list, MINI_DB_PATH)

        reporter.log_chunks_sample(chunks)
        reporter.save()
        print(f"   [OK] Indexado ({len(chunks)} chunks).")

    print("\n✅ Processo Finalizado.\n")

if __name__ == "__main__":
    while True:
        print("\n🎯 RAG PIPELINE - MENU")
        print("[1] ✨ Resetar TUDO")
        print("[2] 📚 Indexar Todos")
        print("[3] ⚙️ Indexar com Metadados")
        print("[0] ❌ Sair")
        op = input("\n👉 Opção: ").strip()
        
        if op == "1":
            if input("Confirma (S/N)? ").upper() == "S": limpar_banco_dados()
        elif op == "2":
            indexar_documentos()
        elif op == "3":
            print("\n🔧 Modo: Indexar com metadados por arquivo")
            indexar_documentos(metadados_padrao={})
        elif op == "0":
            break