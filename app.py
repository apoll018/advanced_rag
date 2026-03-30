"""
Project: Advanced RAG Pipeline
Author: Lucas Henrique da S. Medeiros
GitHub: https://github.com/apoll018
Created: 2026

Este código é de autoria original. Uso sujeito à licença do projeto.
"""

import os
import re
import time
import threading
import subprocess

# ---- Configuráveis ----
DEV_TUNNEL_PORT = int(os.getenv("DEV_TUNNEL_PORT", "8000"))
DEV_TUNNEL_ALLOW_ANON = True  # sempre anônimo
CAPTURE_TIMEOUT_SECONDS = int(os.getenv("CAPTURE_TIMEOUT_SECONDS", "45"))

# Regex para capturar URLs do Dev Tunnel
REGEX_PREFERRED = re.compile(r"(https://[\w\-.]+-8000\.[\w\-.]+\.devtunnels\.ms\S*)")
REGEX_GENERIC   = re.compile(r"(https://[\w\-.]+\.devtunnels\.ms\S*)")

def sanitize_url(url: str) -> str:
    """Remove ':8000' do host se presente."""
    return re.sub(r"(?<=devtunnels\.ms):8000(?=[/\s]|$)", "", url)

def ler_saida_tunnel(processo, url_container):
    """Lê stdout do túnel até achar uma URL pública."""
    try:
        for linha in iter(processo.stdout.readline, ""):
            if not linha:
                break
            line = linha.strip()
            print(f"[TUNNEL LOG]: {line}")

            # Regex preferida primeiro
            m = REGEX_PREFERRED.search(line)
            if not m:
                m = REGEX_GENERIC.search(line)

            if m:
                url_container['url'] = sanitize_url(m.group(1))
                print(f"\n✅ URL CAPTURADA: {url_container['url']}\n")
                return
    except Exception as e:
        print(f"[TUNNEL ERROR]: {e}")

def start_devtunnel():
    """Inicia o Dev Tunnel anônimo na porta definida."""
    tunnel_cmd = ["devtunnel", "host", "-p", str(DEV_TUNNEL_PORT), "--allow-anonymous"]
    print(f"⏳ Abrindo Dev Tunnel: {' '.join(tunnel_cmd)}")
    proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    return proc

def main():
    print("🚀 Iniciando automação...")

    # 1) Inicia Dev Tunnel
    tunnel_process = start_devtunnel()

    # 2) Captura a URL
    url_data = {}
    reader_thread = threading.Thread(target=ler_saida_tunnel, args=(tunnel_process, url_data), daemon=True)
    reader_thread.start()

    deadline = time.time() + CAPTURE_TIMEOUT_SECONDS
    while time.time() < deadline and 'url' not in url_data:
        if tunnel_process.poll() is not None:
            print("❌ Dev Tunnel encerrou prematuramente.")
            break
        time.sleep(0.2)

    if 'url' not in url_data:
        print("❌ Não foi possível capturar a URL do túnel.")
        tunnel_process.terminate()
        tunnel_process.wait()
        exit(1)

    public_url = url_data['url']
    os.environ["PUBLIC_URL"] = public_url
    print(f"🌐 PUBLIC_URL definida: {public_url}")

    # 3) Inicia FastAPI
    import uvicorn
    module_name = os.getenv("FASTAPI_APP", "fastAPImain:app")
    try:
        uvicorn.run(module_name, host="0.0.0.0", port=DEV_TUNNEL_PORT, reload=False)
    finally:
        print("💀 Fechando túnel...")
        tunnel_process.terminate()
        tunnel_process.wait()

if __name__ == "__main__":
    main()
