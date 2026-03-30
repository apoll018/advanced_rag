"""
Módulo de Geração de Resposta (LLM)
-----------------------------------
Responsável por transformar o contexto recuperado em uma resposta final.
Utiliza exclusivamente o Google Gemini (Via Client Nativo REST - Blindado contra Proxy).
"""

import os
import logging
import sys
from typing import Tuple
from dotenv import load_dotenv

# --- Imports de IA ---
from google import genai # Client Nativo (Mais seguro para redes corporativas)

# --- Configuração de Caminhos ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

ENV_PATH = os.path.join(BASE_DIR, "data", ".env")
load_dotenv(dotenv_path=ENV_PATH)

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 📝 PROMPT PADRÃO (Markdown Otimizado)
# ==========================================

PROMPT_TEMPLATE = """
Você é um Especialista Sênior em Normas, Processos e Compliance do Grupo CCR.
Sua tarefa é responder de forma completa, clara e totalmente baseada nos documentos fornecidos como CONTEXTO.

====================================================
🎨 REGRAS DE FORMATAÇÃO (VISUAL)
====================================================
1. Use **Markdown** para estruturar a resposta.
2. Destaque em **negrito** todos os valores monetários (ex: **R$ 1.200,00**), prazos e nomes de documentos.
3. Use listas (bullets) para enumerar regras ou condições.
4. Se houver comparação de valores, você pode usar tabelas Markdown.
5. Não use Cabeçalhos H1 (#) ou H2 (##). Use apenas H3 (###) para os títulos das seções.

====================================================
🎯 COMO RESPONDER (CONTEÚDO)
====================================================
1. **Leia TODO o contexto.** Identifique regras, exceções e valores numéricos.
2. **Seja Exaustivo.** Nunca dê respostas curtas. Explique o "porquê" e o "como".
3. **Honestidade.** Se a informação não estiver no contexto, diga: "Informação não presente nos documentos fornecidos."
4. **Fidelidade.** Não invente dados.

====================================================
📌 ESTRUTURA DA RESPOSTA
====================================================

### Resposta Direta
Forneça a resposta objetiva e direta. Destaque o valor ou a regra principal em negrito.

### Detalhamento Completo
Explique a regra com profundidade:
- Defina os termos técnicos envolvidos.
- Descreva como o limite/regra funciona na prática.
- Cite condições importantes.

### Exceções, Regras Relacionadas ou Limites Próximos
Liste (usando bullets) outras regras do contexto que:
- Podem causar dúvida (ex: limites para outros cargos ou situações).
- Se relacionam com a categoria perguntada.
*Se não houver exceções, diga apenas: "Não foram encontradas exceções relevantes no contexto."*

### Referências
Liste as normas usadas no formato (NOR_XXX > Seção Y).

====================================================
📄 CONTEXTO:
{contexto}

🧭 PERGUNTA:
{pergunta}
"""

# ==========================================
# ☁️ MOTOR: GOOGLE GEMINI (NATIVO REST)
# ==========================================

def _validar_google_key() -> Tuple[bool, str]:
    key = os.environ.get("GEMINI_API_KEY") 
    if not key:
        # Tenta GOOGLE_API_KEY como fallback
        key = os.environ.get("GOOGLE_API_KEY")
    
    if not key or len(key) < 20:
        return False, "❌ Erro: GEMINI_API_KEY não encontrada no .env."
    return True, key

from google import genai

def _gerar_via_gemini(pergunta: str, contexto: str) -> str:
    ok, key_or_msg = _validar_google_key()
    if not ok:
        return key_or_msg

    api_key = key_or_msg

    try:
        logger.info("☁️ Conectando ao Google Gemini (Novo SDK)...")

        # Cliente novo
        client = genai.Client(api_key=api_key)

        # Prompt
        prompt_final = PROMPT_TEMPLATE.format(
            contexto=contexto,
            pergunta=pergunta
        )

        # Geração
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_final,
            config={
                "temperature": 0.1,
                "max_output_tokens": 4000,
            }
        )

        if response.text:
            return response.text
        else:
            return "O modelo não gerou texto."

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Falha no Gemini: {error_msg}")

        if "429" in error_msg:
            return "⚠️ Cota da API excedida. Aguarde um momento."
        if "SSL" in error_msg or "certificate" in error_msg.lower():
            return "⚠️ Erro de Bloqueio de Rede (SSL)."

        return f"Erro na geração via Nuvem: {error_msg}"

# ==========================================
# 🚀 ORQUESTRADOR (FACADE)
# ==========================================

def gerar_resposta_llm(pergunta: str, contexto: str) -> str:
    """
    Função principal para geração de resposta.
    """
    if not pergunta or not contexto:
        return "❌ Erro: Pergunta ou Contexto vazios."

    return _gerar_via_gemini(pergunta, contexto)