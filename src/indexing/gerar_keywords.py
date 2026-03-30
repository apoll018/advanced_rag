"""
Project: Advanced RAG Pipeline
Author: Lucas Henrique da S. Medeiros
GitHub: https://github.com/apoll018
Created: 2026

Este código é de autoria original. Uso sujeito à licença do projeto.
"""

import json
from openai import OpenAI
import time

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def gerar_keywords(texto: str, max_tentativas: int = 2) -> list:
    """
    Gera palavras-chave com retry e tratamento de erros melhorado.
    """
    if not texto or len(texto.strip()) < 30:
        return []
    
    texto_limitado = texto[:3000]
    
    prompt = f"""
    Você deve analisar o documento abaixo, que é uma Norma, Política ou Procedimento Operacional do Grupo CCR.

    INSTRUÇÕES OBRIGATÓRIAS:

    1. Ignore completamente tudo que estiver nas seções iniciais, como:
    - Objetivo
    - Aprovação e Vigência
    - Aplicação
    - Referências
    - Definições

    2. Comece a identificar palavras-chave SOMENTE a partir das seções que contenham:
    - "Disposições Gerais"
    - "Diretrizes"
    - "Regras"
    - "Procedimentos"
    - "Processos"
    - "Orientações"
    - "Critérios"
    - "Etapas"
    - ou qualquer seção similar que represente o conteúdo normativo aplicado.

    3. Extraia **ATÉ 15 palavras-chave**, seguindo estas regras:
    - Use apenas palavras isoladas (nunca frases).
    - Não repita palavras ou termos com o mesmo significado.
    - Não invente termos que não existam no documento.
    - Extraia somente temas essenciais da norma, evitando:
        • nomes de cargos
        • nomes de áreas
        • termos definidos na seção “Definições”
        • números, siglas ou abreviações técnicas irrelevantes

    4. Sua resposta deve ser EXCLUSIVAMENTE um array JSON válido. Nada além disso.
    Não escreva explicações, frases ou comentários antes ou depois do JSON.

    Exemplo de formato correto:
    ["palavra1", "palavra2", "palavra3"]
    
    RETORNE APENAS O ARRAY JSON.

    DOCUMENTO:
    {texto_limitado}
        """

    for tentativa in range(max_tentativas):
        try:
            print(f"   [Tentativa {tentativa + 1}] Gerando keywords...")
            
            resp = client.chat.completions.create(
                model="qwen3:1.7b-q4_K_M",
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            # ✅ VERIFICAÇÃO: content pode ser None
            if not resp.choices or not resp.choices[0].message.content:
                print(f"   [Resposta vazia] Ollama retornou None")
                if tentativa < max_tentativas - 1:
                    time.sleep(2)
                    continue
                return []
            
            conteudo = resp.choices[0].message.content.strip()
            
            # Tenta decodificar JSON
            keywords = json.loads(conteudo)
            
            if not isinstance(keywords, list):
                keywords = [keywords] if keywords else []
            
            keywords = [k.strip() for k in keywords if isinstance(k, str) and len(k.strip()) > 2]
            
            return list(set(keywords))[:12]
            
        except json.JSONDecodeError:
            print(f"   [JSON Error] Resposta inválida: {conteudo[:100]}")
            continue
        except Exception as e:
            print(f"   [Erro Tentativa {tentativa + 1}] {type(e).__name__}: {str(e)[:100]}")
            if tentativa < max_tentativas - 1:
                time.sleep(2)
                continue
            break
    
    return []