# main.py

"""
Project: Advanced RAG Pipeline
Author: Lucas Henrique da S. Medeiros
GitHub: https://github.com/apoll018
Created: 2026

Este código é de autoria original. Uso sujeito à licença do projeto.
"""

import os
import ssl
import urllib3
import subprocess

STREAMLIT_APP_PATH = "src/app/streamlit_app.py"

def run_streamlit():
    """Executa a aplicação Streamlit usando o comando 'streamlit run'."""
    print("==================================================")
    print("         INICIANDO PIPELINE RAG AVANÇADO         ")
    print("==================================================")

    if not os.path.exists(STREAMLIT_APP_PATH):
        print(f"ERRO: Arquivo Streamlit não encontrado em: {STREAMLIT_APP_PATH}")
        print("Verifique o caminho e a estrutura de pastas.")
        return

    # Comando para rodar o Streamlit
    command = ["streamlit", "run", STREAMLIT_APP_PATH]

    try:
        # Executa o comando e mantém o terminal bloqueado
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\nERRO: O comando 'streamlit' não foi encontrado.")
        print("Certifique-se de que o Streamlit está instalado (pip install streamlit) e no PATH.")
    except Exception as e:
        print(f"\nErro ao rodar o Streamlit: {e}")

if __name__ == "__main__":
    # Carrega as variáveis de ambiente antes de tudo (opcional, mas boa prática)
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='data/.env')
    
    run_streamlit()
    