"""
Módulo de Reescrita de Consulta (Query Rewriter)
------------------------------------------------
Responsável por refinar a pergunta do usuário antes da busca.

Modo Atual: ULTRA-RÁPIDO (Google Gemini API)
- Usa SDK nativo do Google Generative AI.
- Modelo: Gemini 1.5 Flash (Baixa latência).
- Enriquecimento Semântico (Mini-DB): DESATIVADO [MODO TESTE].
"""

import os
import sys
from typing import List, Tuple, Any
from dotenv import load_dotenv
from google import genai as genai 

# --- Configuração ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

ENV_PATH = os.path.join(BASE_DIR, "data", ".env")
load_dotenv(dotenv_path=ENV_PATH)
MINI_DB_PATH = os.path.join(BASE_DIR, "data", "mini_vector_db.pkl")

from indexing.mini_vector_db import carregar_mini_db, consultar_mini_db

# ==========================================
# ⚡ CLIENTE GEMINI (GOOGLE AI)
# ==========================================

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") # ou GOOGLE_API_KEY dependendo do seu .env

# Modelo Ultra-Rápido para reescrita
MODELO_REESCRITA = "gemini-2.5-flash"

_gemini_configured = False
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        _gemini_configured = True
    except Exception as e:
        print(f"[WARN] Erro ao configurar Gemini: {e}")

# Configuração de geração para garantir respostas curtas e diretas
generation_config = {
    "temperature": 0.1,
    "max_output_tokens": 128,
    "response_mime_type": "text/plain",
}

REWRITE_PROMPT = """\
Você é um assistente de busca especializado em recuperar informações.
Sua tarefa: Reescrever a pergunta do usuário para torná-la gramaticalmente correta, direta e otimizada para busca vetorial.

Regras RÍGIDAS:
1. NÃO responda a pergunta.
2. NÃO adicione explicações, introduções ou "Aqui está a reescrita".
3. Mantenha termos técnicos intactos.
4. Retorne APENAS o texto da pergunta reescrita.

Pergunta Original: "{question}"
Termos de Apoio (Opcional): "{terms}"

Reescrita:"""

def rewrite_query(user_query: str, terms: str = "") -> str:
    """
    Executa a reescrita via Google Gemini API (Flash).
    """
    if not _gemini_configured:
        return user_query

    try:
        # Instancia o modelo
        model = genai.GenerativeModel(
            model_name=MODELO_REESCRITA,
            generation_config=generation_config
        )
        
        prompt = REWRITE_PROMPT.format(question=user_query, terms=terms)

        # Chamada síncrona ao Gemini
        response = model.generate_content(prompt)

        # Verificação de segurança e conteúdo vazio
        if not response.text:
            return user_query

        consulta_reescrita = response.text.strip()
        
        # Remove aspas ou quebras de linha indesejadas
        return consulta_reescrita.replace('"', '').replace('\n', ' ')

    except Exception as e:
        # Captura erros de quota, rede ou bloqueio de segurança
        print(f"[WARN] Falha na reescrita Gemini ({e}). Usando original.")
        return user_query

# ==========================================
# 🔍 LÓGICA DE ENRIQUECIMENTO (DESATIVADA)
# ==========================================

def identificar_documentos_relevantes(
    query: str,
    emb: Any,
    top_k_docs: int = 3
) -> Tuple[List[str], List[str]]:
    
    try:
         mini_db = carregar_mini_db(MINI_DB_PATH)
         top_termos, _doc_scores, top_docs_scores = consultar_mini_db(
             query=query,
             mini_db=mini_db,
             emb=emb,
             top_k_docs=top_k_docs
         )
         normas_identificadas = [doc for doc, _score in top_docs_scores]
         print(f"[ROUTER] Normas ativadas: {normas_identificadas}")
         return top_termos, normas_identificadas
    except Exception as e:
         print(f"[ROUTER] Erro ou Mini-DB não encontrado ({e}).")
         return [], []


# ==========================================
# 🚀 CONTROLLER
# ==========================================

def preparar_consulta(user_query: str, embedding_model: Any) -> Tuple[str, List[str]]:
    """
    Pipeline Ajustado (Modo Offline/Mock):
    1. Consulta Mini-DB (Router) normalmente.
    2. PULA a reescrita do Gemini (evita erro de rede).
    3. Retorna query original + lista de documentos para filtro.
    """
    
    # 1. Router Ativado (Busca documentos e termos)
    _termos_list, docs_nomes = identificar_documentos_relevantes(user_query, embedding_model)

    # 2. Reescrita (DESATIVADA TEMPORARIAMENTE)
    # Como estamos sem rede para o Gemini, usamos a query original.
    # Se a rede voltar, basta descomentar a linha abaixo:
    
    # consulta_reescrita = rewrite_query(user_query, terms=", ".join(termos_list))
     
    consulta_reescrita = rewrite_query(user_query)
    
    print("-" * 40)
    print(f"[QUERY] Original:  '{user_query}'")
    print(f"[QUERY] Reescrita: '{consulta_reescrita}' (Gemini OFF)")
    print(f"[ROUTER] Filtro:    {docs_nomes if docs_nomes else 'GLOBAL (Sem filtro)'}")
    print("-" * 40)

    return consulta_reescrita, docs_nomes