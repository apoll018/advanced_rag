# splitters.py

import os
import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# CONFIGURAÇÃO
# ============================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))

# ============================================================
# LIMPEZA LEVE (SAFE MODE)
# ============================================================

def limpar_texto(text: str) -> str:
    """
    Limpeza leve para normalizar espaçamentos e bullets.
    Não remove conteúdo, apenas normaliza estrutura.
    """

    # Normaliza bullets do PDF
    text = re.sub(r"[\n\r ]*•[ ]*", "\n- ", text)

    # Normaliza múltiplas linhas vazias
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove espaços no fim da linha
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    return text.strip()

# ============================================================
# SPLITTER ÚNICO (SMALL-TO-BIG LOCAL)
# ============================================================

def get_main_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ",
            "\n### ",
            "\n- ",
            "\n\n",
            ". ",
            " ",
            ""
        ]
    )

# ============================================================
# GERADOR DE CHUNKS
# ============================================================

def gerar_chunks(texto: str) -> List[str]:
    """
    Recebe texto já limpo e aplica chunk único.
    """
    texto = limpar_texto(texto)
    splitter = get_main_splitter()
    return splitter.split_text(texto)
