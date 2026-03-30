from typing import List
from langchain_core.documents import Document

def format_docs(docs: List[Document]) -> str:
    """
    Converte os documentos recuperados em uma string estruturada para a LLM.
    
    Formato de Saída:
    --- DOCUMENTO 1 ---
    FONTE: NOR 007 > 1. OBJETIVO (Pág. 2)
    CONTEÚDO:
    Texto do documento...
    
    --- DOCUMENTO 2 ---
    ...
    """
    if not docs:
        return ""

    contexto_formatado = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        
        # Recupera dados com fallback seguro
        nome_doc = meta.get('documento_completo', 'Documento Desconhecido')
        hierarquia = meta.get('hierarquia', 'Seção Geral')
        pagina = meta.get('page', 'N/A')

        # Monta a linha de referência completa
        # Ex: NOR 007 > 6.2.2 (Pág. 5)
        fonte_completa = f"{nome_doc} > {hierarquia}"
        if pagina != 'N/A':
            fonte_completa += f" (Pág. {pagina})"

        # Bloco formatado com delimitadores visuais claros
        bloco = (
            f"--- DOCUMENTO {i} ---\n"
            f"FONTE: {fonte_completa}\n"
            f"CONTEÚDO:\n{doc.page_content.strip()}\n"
        )

        contexto_formatado.append(bloco)

    # Usa quebra dupla para separar bem os blocos
    return "\n\n".join(contexto_formatado)