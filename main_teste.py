import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from generation.llm_response import gerar_resposta_llm
try:
    from src.app.rag_pipeline import run_rag_pipeline_hibrido
except ImportError as e:
    print(f"\n❌ Erro de Importação: {e}")
    print("Verifique se o ambiente virtual (.venv) está ativo.")
    sys.exit(1)

def main():
    print("\n" + "=" * 60)
    print("⚡ MODO DE TESTE RÁPIDO INTERATIVO - SIRIS")
    print("Digite 'sair', 'quit' ou 'exit' para encerrar.")
    print("=" * 60)

    while True:
        # Recebe a pergunta do usuário via terminal
        pergunta = input("\nDigite sua pergunta (ou 'sair'): ").strip()

        # Condição de parada
        if pergunta.lower() in ['sair', 'quit', 'exit']:
            print("\nEncerrando o terminal de testes. Até logo! 👋\n")
            break

        if not pergunta:
            continue

        print("⏳ Processando... (Buscando normas e gerando resposta)")
        start_time = time.time()

        try:
            # 1. Executa a Recuperação (RAG Híbrido + Reranker)
            resultado = run_rag_pipeline_hibrido(pergunta)
            docs = resultado.get("docs_ranqueados", [])
            
            # Extrai apenas os textos para enviar ao LLM
            retrieved_contexts = [doc.page_content for doc in docs]

            # 2. Gera a Resposta com o Gemini
            resposta_llm = gerar_resposta_llm(pergunta, retrieved_contexts)
            
            duration = time.time() - start_time

            # ======================================
            # EXIBIÇÃO NO CONSOLE
            # ======================================
            print("\n" + "=" * 60)
            print("🤖 RESPOSTA DO SIRIS:")
            print("=" * 60)
            print(resposta_llm)
            print("\n" + "-" * 60)
            
            # Exibe os metadados dos chunks usados para auditar a resposta
            print(f"📚 Documentos recuperados pelo Reranker: {len(docs)}")
            for i, doc in enumerate(docs):
                meta = doc.metadata
                arquivo = meta.get("documento_completo", "N/A")
                score = meta.get("score_rerank", 0)
                print(f"   📄 [Rank {i+1}] {arquivo} (Score: {score:.4f})")
                
            print(f"⏱️ Tempo total da operação: {duration:.2f}s")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ ERRO AO PROCESSAR A PERGUNTA: {e}")

if __name__ == "__main__":
    main()