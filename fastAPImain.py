from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, sys, os
from typing import List
import traceback 
import re
# Configuração de Caminhos (Path)

# Desliga completamente a telemetria do Chroma
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from src.app.rag_pipeline import run_rag_pipeline_hibrido
# Inicializa a API
# Isso permite que o script de automação injete a URL aqui
url_servidor = os.getenv("PUBLIC_URL", "https://zfn8f710-8000.brs.devtunnels.ms")

#CRIAR TUNEL: devtunnel host -p 8000 --allow-anonymous

app = FastAPI(
    title="API SIRIS",
    description="Backend conectado ao Microsoft Copilot",
    version="1.0.0",
    servers=[
        {"url": url_servidor, "description": "Servidor Automático"}
    ]
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# Modelo de entrada
class PerguntaUsuario(BaseModel):
    pergunta: str

# Modelo de saída (Ajustado para o que o Copilot precisa ver)
class RespostaRAG(BaseModel):
    contexto_completo: str
    fontes: List[str]
    debug_tempo: str
    mensagem_sistema: str # Dica para o Copilot

@app.post("/consultar-rag", response_model=RespostaRAG)
async def consultar_rag(dados: PerguntaUsuario):
    print(f"🔵 Requisição do Copilot: {dados.pergunta}")
    
    try:
        # 1. Executa o pipeline (Busca Híbrida + Reranker)
        # O pipeline deve retornar os Documentos já ranqueados pelo BGE-M3
        resultado = run_rag_pipeline_hibrido(dados.pergunta)
        docs_ranqueados = resultado.get("docs_ranqueados", [])

        # 2. Prepara o contexto bruto para o Copilot
        # Vamos unir os Top 5 trechos em uma única string clara
        trechos_texto = []
        lista_fontes = []

        for i, doc in enumerate(docs_ranqueados):
            conteudo = doc.page_content
            fonte = doc.metadata.get("documento_completo", "Norma Interna")
            hierarquia = doc.metadata.get("hierarquia", "")
            
            # Monta um bloco de texto identificado para o Copilot não se perder
            bloco = f"--- TRECHO {i+1} (Fonte: {fonte} | {hierarquia}) ---\n{conteudo}\n"
            trechos_texto.append(bloco)
            lista_fontes.append(fonte)

        contexto_para_copilot = "\n".join(trechos_texto)
        lista_fontes = list(set(lista_fontes)) # Remove duplicados

        # 3. Ajuste de Segurança (Token Limit)
        # O Copilot tem um limite. 15k a 18k caracteres é um porto seguro.
        if len(contexto_para_copilot) > 18000:
            contexto_para_copilot = contexto_para_copilot[:18000] + "... [Conteúdo truncado por tamanho]"

        print(f"🟢 Enviando {len(docs_ranqueados)} chunks filtrados para o Copilot.")

        return {
            "contexto_completo": contexto_para_copilot,
            "fontes": lista_fontes,
            "debug_tempo": resultado.get("tempo_total", "0s"),
            "mensagem_sistema": (
                "Você é o assistente SIRIS. Responda APENAS com base nos trechos fornecidos acima. "
                "Se a resposta não estiver nos trechos, diga que não localizou a informação nas normas vigentes. "
                "Sempre cite o nome da Norma (ex: NOR_013) ao fornecer um dado técnico."
            )
        }

    except Exception as e:
        print(f"🔴 Erro no processamento: {e}")
        traceback.print_exc()
        return {
            "contexto_completo": "Erro técnico ao acessar a base de conhecimento da Motiva.",
            "fontes": [],
            "debug_tempo": "0s",
            "mensagem_sistema": "Ocorreu uma falha no backend de busca."
        }


# Isso permite rodar o arquivo direto com 'python main.py'
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #RODAR A API: uvicorn fastApi:app --host 0.0.0.0 --port 8000 