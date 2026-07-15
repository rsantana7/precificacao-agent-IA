"""
Agente de IA Executivo (Chief Revenue Officer) — versão STREAMLIT / STREAMLIT CLOUD.

Usa a API do Google (Gemini) para redigir o parecer estratégico de precificação.
A chave é lida de st.secrets["GOOGLE_API_KEY"] (configurada em
Settings > Secrets no Streamlit Community Cloud, ou em .streamlit/secrets.toml
localmente), com fallback para variável de ambiente.
"""

import os
import streamlit as st
from google import genai
from google.genai import errors as genai_errors

MODEL_NAME = "gemini-3.5-flash"  # troque para "gemini-3.5-flash" se sua conta já tiver acesso


def _obter_api_key():
    """Busca a chave primeiro em st.secrets (Streamlit Cloud) e depois em variável de ambiente."""
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        # st.secrets levanta exceção se não houver nenhum arquivo secrets.toml configurado.
        pass
    return os.getenv("GOOGLE_API_KEY")


def gerar_parecer_precificacao(
    preco_atual, preco_sugerido, lucro_atual, lucro_projetado, concorrente, estoque
):
    """Atua como um Agente de IA Executivo (Revenue Manager) justificando a estratégia de preços."""

    api_key = _obter_api_key()
    if not api_key:
        return (
            "⚠️ **Configuração pendente**\n\n"
            "A chave `GOOGLE_API_KEY` não foi encontrada nos *Secrets* do app.\n\n"
            "No Streamlit Community Cloud: abra **Settings → Secrets** e adicione:\n\n"
            "```toml\nGOOGLE_API_KEY = \"sua_chave_aqui\"\n```\n\n"
            "Localmente, crie o arquivo `.streamlit/secrets.toml` com o mesmo conteúdo.\n\n"
            "Você pode gerar uma chave gratuita em https://aistudio.google.com/apikey"
        )

    prompt_sistema = (
        "Você é um renomado Chief Revenue Officer (CRO) e especialista em Precificação Dinâmica. "
        "Sua função é analisar métricas de elasticidade de preço e redigir um parecer técnico, "
        "curto, persuasivo e altamente analítico para a diretoria executiva, justificando a "
        "alteração de preços."
    )

    prompt_usuario = f"""
    Analise o seguinte cenário de precificação gerado pelo nosso algoritmo preditivo:
    - Preço Atual Praticado: R$ {preco_atual:.2f}
    - Preço Sugerido Otimizado: R$ {preco_sugerido:.2f}
    - Lucro Estimado com Preço Atual: R$ {lucro_atual:.2f}
    - Lucro Projetado Otimizado: R$ {lucro_projetado:.2f}
    - Preço do Concorrente Direto: R$ {concorrente:.2f}
    - Nível de Estoque Atual: {estoque} unidades

    Por favor, gere um relatório executivo estruturado com os seguintes tópicos (use markdown):
    1. **Análise de Elasticidade e Mercado**: Explicar por que a mudança de preço faz sentido frente ao concorrente e estoque.
    2. **Impacto Financeiro**: Destacar o ganho percentual na margem de lucro.
    3. **Plano de Ação / Recomendação**: Uma diretriz clara para o time de marketing/vendas aplicar a alteração imediatamente.
    """

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return (
            "❌ **Não foi possível inicializar o cliente da API Google.**\n\n"
            f"Detalhe técnico: `{e}`"
        )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"{prompt_sistema}\n\n{prompt_usuario}",
        )
    except genai_errors.ClientError as e:
        # Erros 4xx: chave inválida, sem permissão, cota excedida, etc.
        return (
            "🔒 **A API do Google recusou a solicitação.**\n\n"
            "Verifique se a chave configurada em *Secrets* é válida e possui cota disponível.\n\n"
            f"Detalhe técnico: `{e}`"
        )
    except genai_errors.ServerError as e:
        # Erros 5xx: instabilidade temporária no lado do Google.
        return (
            "🌩️ **O serviço da IA está temporariamente indisponível.**\n\n"
            "Isso costuma ser passageiro — tente executar a otimização novamente em instantes.\n\n"
            f"Detalhe técnico: `{e}`"
        )
    except Exception as e:
        # Falhas de rede, timeout, DNS, etc.
        return (
            "📡 **Falha de comunicação com o Agente de IA.**\n\n"
            "Verifique a conectividade do app e tente novamente.\n\n"
            f"Detalhe técnico: `{e}`"
        )

    texto = getattr(response, "text", None)
    if not texto:
        return (
            "⚠️ **O Agente de IA respondeu, mas sem conteúdo de texto utilizável.**\n\n"
            "Tente executar a otimização novamente."
        )

    return texto
