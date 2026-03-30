"""
Project: Advanced RAG Pipeline
Author: Lucas Henrique da S. Medeiros
GitHub: https://github.com/apoll018
Created: 2026

Este código é de autoria original. Uso sujeito à licença do projeto.
"""

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# Configuração de Caminhos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

from app.rag_pipeline import run_rag_pipeline_hibrido

# ============================================================
# ⚙️ CONFIGURAÇÃO VISUAL (CSS DEFINITIVO)
# ============================================================
st.set_page_config(page_title="RAG Corporativo | Compliance", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    /* 1. Correção de Fonte Grande no Chat */
    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3 {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        margin-top: 10px !important;
        margin-bottom: 5px !important;
    }
    
    /* 2. Impedir Overflow (Texto saindo do card) */
    /* Evitar overflow dentro de blocos do chat */
    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span {
        word-wrap: break-word !important;
        overflow-wrap: anywhere !important;
        white-space: normal !important;
    }

    /* Evitar overflow no conteúdo das fontes (expander) */
    .block-container .stCode pre,
    .block-container .stCode code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }


    /* Cards de Etapas */
    .step-card { 
        padding: 8px 12px; 
        margin-bottom: 6px; 
        border-radius: 6px; 
        background-color: #f8f9fa; 
        border: 1px solid #dee2e6;
        font-size: 0.85em;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .step-card.completed { border-left: 4px solid #28a745; }
    .step-card.running { border-left: 4px solid #007bff; }
    .step-card.error { border-left: 4px solid #dc3545; }
    
    /* Badges */
    .score-badge { padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.75em; color: white; }
    .expanded-badge { background-color: #6f42c1; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7em; font-weight: bold; margin-left: 6px; }
</style>
""", unsafe_allow_html=True)

load_dotenv(os.path.join(BASE_DIR, "data", ".env"))

# ============================================================
# 🔧 FUNÇÕES AUXILIARES
# ============================================================

def escape_latex(text: str) -> str:
    """Evita que valores em Reais (R$) sejam interpretados como LaTeX."""
    if not text: return ""
    # Substitui $ por sua entidade HTML ou escapa, mas o jeito mais seguro no Streamlit 
    # é trocar por uma string que não ativa o MathJax, como 'R$ ' com espaço.
    return text.replace("$", "\\$") 

def renderizar_etapas(etapas: list):
    if not etapas: return
    st.markdown("#### ⚡ Execução do Pipeline")
    for step in etapas:
        # Ícones e Status
        if step['status'] == 'completed':
            icon, color_class = "✅", "completed"
        elif step['status'] == 'running':
            icon, color_class = "🔄", "running"
        elif step['status'] == 'error':
            icon, color_class = "❌", "error"
        else:
            icon, color_class = "⏳", "pending"

        # Duração
        dur = f"{step['duration']:.3f}s" if step.get('duration') is not None else "..."
        
        st.markdown(f"""
            <div class='step-card {color_class}'>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#333;">{icon} {step['name']}</strong>
                    <span style="font-family:monospace; color:#555;">{dur}</span>
                </div>
                <div style="color:#666; font-size:0.9em; margin-top:2px;">{step['description']}</div>
            </div>
        """, unsafe_allow_html=True)

def executar_chat(pergunta: str, status_box):
    _start = time.time()
    try:
        # Backend Call
        resultado = run_rag_pipeline_hibrido(pergunta)
        
        # Tratamento de resposta vazia
        resposta = resultado.get('resposta')
        if not resposta: resposta = "Não consegui gerar uma resposta."
        
        # Escape visual para o frontend
        resposta_visual = escape_latex(resposta)

        # Salva estado
        st.session_state['last_run'] = {
            "query_rewrite": resultado.get('pergunta_reescrita'),
            "docs": resultado.get('docs_ranqueados', []),
            "resposta": resposta_visual,
            "etapas": resultado.get('etapas', []),
            "tempo_total": resultado.get('tempo_total', 'N/A'),
            "filtros": resultado.get('documentos_filtrados', []),
            "contexto": resultado.get('contexto_usado', '')
        }
        
        # Fecha status box
        status_box.update(label="✅ Processamento concluído!", state="complete", expanded=False)
        return resposta_visual

    except Exception as e:
        status_box.update(label="Erro crítico", state="error")
        return f"Ocorreu um erro: {str(e)}"

# ============================================================
# 🧭 APP LAYOUT
# ============================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9564/9564496.png", width=60)
    st.title("Compliance RAG")
    st.caption("Sistema de Consulta Normativa Inteligente")
    st.markdown("---")
    st.info("""
    **Versão 2.0 (Small-to-Big)**
    - 🧠 **Router:** Detecção de tema
    - 🔎 **E5-Base:** Busca Semântica
    - ⚖️ **BGE-M3:** Reranking Neural
    - 🌟 **Context:** Expansão Dinâmica
    """)

col_chat, col_raio_x = st.columns([1.3, 1])

# --- CHAT ---
with col_chat:
    st.subheader("💬 Assistente")
    container = st.container(height=650)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar com as normas hoje?"}]

    with container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Digite sua pergunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                status = st.status("Iniciando pipeline...", expanded=True)
                resp = executar_chat(prompt, status)
                
                # Efeito digitação
                box = st.empty()
                texto_acumulado = ""
                for pedaco in resp.split(" "):
                    texto_acumulado += pedaco + " "
                    box.markdown(texto_acumulado + "▌")
                    time.sleep(0.02)
                box.markdown(resp)
                
                st.session_state.messages.append({"role": "assistant", "content": resp})

# --- RAIO-X ---
with col_raio_x:
    st.subheader("🔍 Performance & Lógica")
    
    dados = st.session_state.get('last_run')
    
    if dados:
        tab_metricas, tab_fontes, tab_prompt = st.tabs(["⏱️ Métricas", "📚 Fontes", "🧠 Prompt"])
        
        with tab_metricas:
            c1, c2 = st.columns(2)
            c1.metric("Tempo Total", dados['tempo_total'])
            c2.metric("Docs Recuperados", len(dados['docs']))
            
            if dados['filtros']:
                st.success(f"🎯 **Filtro Router:** {dados['filtros']}")
            else:
                st.info("🌍 **Busca Global** (Sem filtros)")
                
            st.divider()
            renderizar_etapas(dados['etapas'])
            
            st.markdown("---")
            st.caption(f"**Reescrita Gemini:** {dados['query_rewrite']}")

        with tab_fontes:
            st.caption("Ordenado por relevância (BGE-Reranker)")
            for i, doc in enumerate(dados['docs']):
                meta = doc.metadata
                score = float(meta.get("score_rerank", 0))
                bg = "#198754" if score > 0.7 else "#fd7e14" if score > 0.2 else "#6c757d"
                
                with st.expander(f"#{i+1} {meta.get('documento_completo', 'DOC')}", expanded=(i==0)):
                    st.markdown(f"""
                        <span class='score-badge' style='background-color:{bg}'>Score: {score:.4f}</span>
                        {'<span class="expanded-badge">🌟 Expandido</span>' if meta.get('expanded') else ''}
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Seção:** {meta.get('hierarquia', 'Geral')}")
                    
                    # Conteúdo com escape para evitar quebra
                    content = escape_latex(doc.page_content)
                    st.code(content, language="markdown")

        with tab_prompt:
            st.caption("Contexto injetado no LLM:")
            st.code(dados['contexto'], language="text")
            
    else:
        st.info("Aguardando interação...")
        st.image("https://cdn-icons-png.flaticon.com/512/6104/6104865.png", width=100)