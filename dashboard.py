import streamlit as st
import pandas as pd
import numpy as np
import pickle
from market_agent import gerar_parecer_precificacao

st.set_page_config(page_title="IA Dynamic Pricing", page_icon="💰", layout="wide")

st.title("💰 Inteligência de Mercado & Precificação Dinâmica")
st.subheader("Otimização de Margem de Lucro com Machine Learning & Agente de Receita AI")
# st.caption("Versão Streamlit Cloud — chave da API lida de st.secrets")
st.markdown("---")


@st.cache_resource
def carregar_modelos_pricing():
    try:
        with open('pricing_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('pricing_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Arquivos de modelo não encontrados. Execute 'python train_pricing_model.py' no terminal primeiro.")
        return None, None
    except Exception as e:
        st.error(f"❌ Falha inesperada ao carregar os modelos: {e}")
        return None, None


model, scaler = carregar_modelos_pricing()

if model and scaler:
    # Interface dividida: Parâmetros do Mercado vs Painel Analítico
    col_esquerda, col_direita = st.columns([1, 2])

    with col_esquerda:
        st.header("🏪 Variáveis de Mercado")
        preco_atual = st.slider("Preço Atual do Seu Produto (R$)", min_value=70.0, max_value=140.0, value=100.0)
        preco_concorrente = st.slider("Preço do Concorrente (R$)", min_value=70.0, max_value=140.0, value=105.0)
        estoque = st.number_input("Nível de Estoque Disponível", min_value=5, max_value=500, value=120)
        fds = st.selectbox("É final de semana ou feriado sazonado?", options=["Não", "Sim"])

        fds_binario = 1 if fds == "Sim" else 0
        custo_produto = 45.0  # Custo fixo simulado de fabricação/aquisição do produto

        st.markdown("---")
        otimizar_btn = st.button("📈 Executar Otimização de Lucro", type="primary")

        # CSS
        st.markdown("---")
        # CSS caixa cinza com texto destacado
        st.markdown(
        """
        
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
            <span style="color: #ff4b4b;">Desenvolvido por :</span>
            <br>
            <br>
            <span style="color: #000000; font-weight: bold;"> Renata LC Santana</span>
            <br>
            <span>Cientista de Dados | Machine Learning |  IA Generativa | LLMs | RAG | Agentes IA | Agile Project Management</span>
        </div>

        """, 
        unsafe_allow_html=True
        )
        # -- fim CSS

    with col_direita:
        st.header("📊 Painel de Simulação Financeira")

        try:
            entrada_atual = np.array([[preco_atual, preco_concorrente, estoque, fds_binario]])
            entrada_atual_scaled = scaler.transform(entrada_atual)
            vendas_atuais_est = max(1, int(model.predict(entrada_atual_scaled)[0]))
            lucro_atual_est = vendas_atuais_est * (preco_atual - custo_produto)
        except Exception as e:
            st.error(f"❌ Não foi possível calcular o cenário atual com o modelo de ML: {e}")
            st.stop()

        if not otimizar_btn:
            # Estado inicial da tela (Exibe dados atuais do negócio)
            c1, c2, c3 = st.columns(3)
            c1.metric("Vendas Estimadas (Volume)", f"{vendas_atuais_est} un")
            c2.metric("Preço Praticado", f"R$ {preco_atual:.2f}")
            c3.metric("Lucro Bruto Estimado", f"R$ {lucro_atual_est:.2f}")
            st.info("💡 Altere os parâmetros à esquerda e clique em **Executar Otimização** para ativar o algoritmo de Machine Learning e o parecer da IA.")

        if otimizar_btn:
            try:
                # Algoritmo de Busca de Preço Ótimo (Simula múltiplos preços em tempo real para achar o maior lucro)
                precos_testados = np.linspace(70, 140, 71)
                melhor_preco = preco_atual
                max_lucro_projetado = 0
                melhor_vendas = vendas_atuais_est

                for p_teste in precos_testados:
                    entrada_teste = np.array([[p_teste, preco_concorrente, estoque, fds_binario]])
                    entrada_teste_scaled = scaler.transform(entrada_teste)
                    vendas_testadas = max(1, int(model.predict(entrada_teste_scaled)[0]))
                    lucro_testado = vendas_testadas * (p_teste - custo_produto)

                    if lucro_testado > max_lucro_projetado:
                        max_lucro_projetado = lucro_testado
                        melhor_preco = p_teste
                        melhor_vendas = vendas_testadas
            except Exception as e:
                st.error(f"❌ Falha ao rodar a otimização de preço com o modelo de ML: {e}")
                st.stop()

            # Exibição Comparativa de Resultados
            c1, c2, c3 = st.columns(3)
            c1.metric(label="Preço Recomendado", value=f"R$ {melhor_preco:.2f}",
                      delta=f"R$ {melhor_preco - preco_atual:.2f} de ajuste")

            ganho_lucro = max_lucro_projetado - lucro_atual_est
            pct_ganho = (ganho_lucro / max(1, lucro_atual_est)) * 100

            c2.metric(label="Lucro Máximo Projetado", value=f"R$ {max_lucro_projetado:.2f}",
                      delta=f"+{pct_ganho:.1f}% de margem" if ganho_lucro > 0 else "Estabilizado")

            c3.metric(label="Previsão de Demanda", value=f"{melhor_vendas} un/período")

            st.markdown("---")
            st.markdown("### 🤖 Parecer Estratégico do Agente de IA (Revenue Manager)")

            with st.spinner("O Agente de IA está consolidando o relatório de posicionamento macroeconômico..."):
                relatorio_ia = gerar_parecer_precificacao(
                    preco_atual, melhor_preco, lucro_atual_est, max_lucro_projetado, preco_concorrente, estoque
                )

            st.markdown(relatorio_ia)
