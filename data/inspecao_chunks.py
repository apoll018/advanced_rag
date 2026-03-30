"""
Ferramenta de Inspeção de Chunks (Visualizador de Cortes)
---------------------------------------------------------
Este script simula o processo de ingestão, mas em vez de salvar no banco,
gera relatórios visuais em Markdown para validar a estratégia de quebra (Splitting).

Objetivo: Verificar se tabelas, listas e parágrafos cruciais (ex: valores monetários)
estão sendo preservados dentro do mesmo chunk ou se estão sendo cortados.
"""

import os
import sys
import shutil
import re
import uuid
from datetime import datetime
from typing import List

# Bibliotecas de processamento
from markitdown import MarkItDown
from pdf2docx import Converter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import unicodedata

# ==========================================
# ⚙️ CONFIGURAÇÕES
# ==========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PDFS_DIR = r"C:\Users\51-02429\Desktop\RAG TCC\data\pdfs"
OUTPUT_DIR = os.path.join(BASE_DIR, "logs", "inspecao_chunks")

# Configuração idêntica ao Splitter de Produção
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Garante que a pasta de saída existe e está limpa
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 🧹 LIMPEZA DE TEXTO (Cópia exata do Indexador)
# ==========================================
def remover_cabecalhos_administrativos(text: str) -> str:
    chaves = [
        r"\|\s*\*\*Norma\*\*", r"\|\s*\*\*Código\*\*", r"\|\s*\*\*Área\*\*", 
        r"\|\s*\*\*Responsável\*\*", r"\|\s*\*\*Vigência.*?\*\*", 
        r"\|\s*\*\*Classificação\*\*", r"\|\s*Norma de Doações"
    ]
    pattern = re.compile(r'^.*(?:' + '|'.join(chaves) + r').*$', re.MULTILINE | re.IGNORECASE)
    return re.sub(pattern, '', text)

def dissolver_tabelas_falsas(text: str) -> str:
    lines = text.split('\n')
    new_lines = []
    pattern = re.compile(r'^\|\s*(\([ivxlcdm]+\)|\d+(\.\d+)*\.?|\*\*\d+.*?|[•\-])\s*\|?', re.IGNORECASE)
    skip = False
    for line in lines:
        s = line.strip()
        if skip:
            if re.match(r'^\|\s*[-:]+\s*\|', s): skip = False; continue
            skip = False
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
    
    # Limpeza visual
    pattern_linha_lixo = re.compile(r'^[ \t\|\.\-_]*$', re.MULTILINE)
    text = re.sub(pattern_linha_lixo, '', text)
    pattern_sequencia = re.compile(r'(?:[•\.\-_]\s*){4,}')
    text = re.sub(pattern_sequencia, ' ', text)
    
    text = re.sub(r'^\s*[|: -]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|\s*$', '', text, flags=re.MULTILINE)
    
    # Headers e formatação
    text = re.sub(r'(\([ivxlcdm]+\).*?;)\s*(?=\(\s*[ivxlcdm]+\s*\))', r'\1\n', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\*\*(\d+)\s+(\d+)\.?\*\*\s*', r'### \1.\2. ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\*(\d+)\.?\*\*\s*(.*)', r'## \1. \2', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\*(\d+(\.\d+)+)\.?\*\*\s*(.*)', r'### \1 \3', text, flags=re.MULTILINE)
    text = re.sub(r'^(#+\s+.*?)\*\*(.*?)\*\*', r'\1\2', text, flags=re.MULTILINE)
    
    text = re.sub(r'^\s*(Página\s*)?\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ==========================================
# 📄 PARSER (Simulado)
# ==========================================

def converter_pdf_markdown(pdf_path: str) -> str:
    print(f"   🔨 Convertendo: {os.path.basename(pdf_path)}...")
    temp_dir = "temp_inspect"
    os.makedirs(temp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    docx_path = os.path.join(temp_dir, f"{base_name}.docx")
    
    if not os.path.exists(docx_path):
        try:
            cv = Converter(pdf_path)
            cv.convert(docx_path, start=0, end=None, parse_header_footer=True)
            cv.close()
        except Exception:
            return ""

    try:
        md = MarkItDown()
        result = md.convert(docx_path)
        raw = result.text_content
        clean = limpar_output_markitdown(raw)
    except Exception:
        clean = ""
    
    try: shutil.rmtree(temp_dir)
    except: pass
    
    return clean

def simular_quebra_chunks(texto_limpo: str) -> List[Document]:
    """
    Aplica a lógica exata de produção para quebrar os chunks.
    """
    # 1. Split Semântico (Headers)
    splitter_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False
    )
    docs_md = splitter_md.split_text(texto_limpo)
    
    # 2. Split Recursivo (Tamanho Fixo)
    # IMPORTANTE: Usando os valores definidos no início do script
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    final_docs = []
    
    for d in docs_md:
        # Lógica de segurança: só quebra se for muito grande
        if len(d.page_content) > (CHUNK_SIZE * 1.2): 
            chunks_rec = rec_splitter.split_text(d.page_content)
            final_docs.extend([
                Document(page_content=t, metadata=d.metadata.copy()) 
                for t in chunks_rec
            ])
        else:
            final_docs.append(d)
            
    return final_docs

# ==========================================
# 📝 GERADOR DE RELATÓRIO
# ==========================================
def gerar_relatorio_arquivo(pdf_name: str, chunks: List[Document]):
    filename = f"INSPECT_{os.path.splitext(pdf_name)[0]}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    buffer = []
    buffer.append(f"# Inspeção de Chunks: {pdf_name}\n")
    buffer.append(f"**Data:** {datetime.now()}\n")
    buffer.append(f"**Total de Chunks:** {len(chunks)}\n")
    buffer.append(f"**Configuração:** Size={CHUNK_SIZE} | Overlap={CHUNK_OVERLAP}\n")
    buffer.append("---\n")
    
    for i, doc in enumerate(chunks):
        h1 = doc.metadata.get("h1", "").replace("#", "").strip()
        h2 = doc.metadata.get("h2", "").replace("#", "").strip()
        h3 = doc.metadata.get("h3", "").replace("#", "").strip()
        hierarquia = " > ".join(filter(None, [h1, h2, h3])) or "Geral"
        
        tamanho = len(doc.page_content)
        
        buffer.append(f"\n## Chunk {i} (Seq: {i})\n")
        buffer.append(f"- **Hierarquia:** `{hierarquia}`\n")
        buffer.append(f"- **Tamanho:** {tamanho} caracteres\n")
        
        # Conteúdo em bloco de código para facilitar leitura
        buffer.append("\n```markdown")
        buffer.append(doc.page_content)
        buffer.append("```\n")
        buffer.append("---\n")
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(buffer))

# ==========================================
# 🚀 MAIN
# ==========================================
def main():
    print("="*60)
    print("🔎 INSPEÇÃO DE CHUNKS E ESTRUTURA")
    print(f"📂 Lendo de: {PDFS_DIR}")
    print(f"💾 Salvando em: {OUTPUT_DIR}")
    print("="*60)
    
    if not os.path.exists(PDFS_DIR):
        print("❌ Pasta de PDFs não encontrada.")
        return

    pdfs = [f for f in os.listdir(PDFS_DIR) if f.lower().endswith(".pdf")]
    print(f"📄 Encontrados {len(pdfs)} arquivos para análise.\n")
    
    for pdf in pdfs:
        print(f"👉 Processando: {pdf}")
        
        # 1. Converte
        texto_limpo = converter_pdf_markdown(os.path.join(PDFS_DIR, pdf))
        if not texto_limpo:
            print("   ❌ Falha na conversão.")
            continue
            
        # 2. Quebra
        chunks = simular_quebra_chunks(texto_limpo)
        
        # 3. Relatório
        gerar_relatorio_arquivo(pdf, chunks)
        print(f"   ✅ Gerado relatório com {len(chunks)} chunks.")

    print("\n🏁 Inspeção concluída! Verifique a pasta 'logs/inspecao_chunks'.")

if __name__ == "__main__":
    main()