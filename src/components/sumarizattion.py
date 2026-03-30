"""
Project: Advanced RAG Pipeline
Author: Lucas Henrique da S. Medeiros
GitHub: https://github.com/apoll018
Created: 2026

Este código é de autoria original. Uso sujeito à licença do projeto.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import os
import torch


def load_local_summarizer(model_path="./src/components/models/t5-legal-summarizer"):
    """
    Carrega o modelo local de sumarização (jurídico, em português).
    Se o modelo não existir localmente, apenas alerta.
    """
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo não encontrado em {model_path}, verifique o caminho.")
    
    print(f"💾 Carregando modelo local de sumarização: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 🚀 Cria o pipeline com parâmetros de geração controlados
    text2text_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=250,          # limite do resumo (em tokens, não caracteres)
        min_length=80,           # garante um resumo mais completo
        repetition_penalty=2.5,  # evita repetições
        length_penalty=1.8,      # penaliza resumos longos
        num_beams=5,             # beam search para maior coerência
        no_repeat_ngram_size=3,  # evita repetições de n-gramas
        do_sample=False,
        truncation=True
    )

    print("✅ Modelo de sumarização carregado com sucesso!")
    return HuggingFacePipeline(pipeline=text2text_pipe)


# =============================================================
# 🧪 Teste rápido (executado apenas se rodar este arquivo direto)
# =============================================================
if __name__ == "__main__":
    summarizer = load_local_summarizer()

    texto = """
A norma de reembolso do Grupo CCR define que todos os colaboradores devem seguir procedimentos padronizados
para solicitar reembolso de despesas com viagens corporativas. As passagens aéreas devem ser emitidas
exclusivamente por agências credenciadas e aprovadas pela área de Suprimentos, com antecedência mínima
de sete dias úteis. Em casos excepcionais, a compra direta poderá ser autorizada pelo gestor imediato,
desde que devidamente justificada. O reembolso de hospedagem será limitado ao valor máximo previsto
na política de viagens vigente, sendo obrigatória a apresentação de nota fiscal em nome da empresa.
"""

    print("\n🧾 Texto original:")
    print(texto)

    resumo = summarizer.invoke(texto)
    print("\n📝 Resumo gerado:")
    print(resumo)
