import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle


def gerar_dados_vendas():
    """Gera dados sintéticos de vendas e elasticidade de preço para e-commerce."""
    np.random.seed(42)
    n_registros = 2000

    preço_praticado = np.random.uniform(70, 140, n_registros)
    preço_concorrente = np.random.uniform(80, 130, n_registros)
    estoque = np.random.randint(5, 500, n_registros)
    fim_de_semana = np.random.choice([0, 1], n_registros, p=[0.7, 0.3])

    # Lógica de elasticidade: preços altos e concorrentes baratos reduzem as vendas. Fim de semana aumenta.
    demanda_base = 200
    vendas = (demanda_base
              - (preço_praticado * 1.5)
              + (preço_concorrente * 0.8)
              + (fim_de_semana * 30)
              - (estoque * 0.05))

    # Garantir que não haja vendas negativas
    vendas = np.clip(vendas, 5, 500).astype(int)

    df = pd.DataFrame({
        'PrecoPraticado': preço_praticado,
        'PrecoConcorrente': preço_concorrente,
        'Estoque': estoque,
        'FimDeSemana': fim_de_semana,
        'QuantidadeVendas': vendas
    })
    return df


def treinar_modelo_precificacao():
    try:
        print("⏳ Coletando dados e treinando modelo de regressão...")
        df = gerar_dados_vendas()

        X = df[['PrecoPraticado', 'PrecoConcorrente', 'Estoque', 'FimDeSemana']]
        y = df['QuantidadeVendas']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Random Forest Regressor - excelente para capturar relações não-lineares de elasticidade de preço
        model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
        model.fit(X_train_scaled, y_train)

        with open('pricing_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('pricing_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        print("✅ Modelo de Precificação e Scaler exportados com sucesso!")
    except Exception as e:
        print(f"❌ Falha ao treinar/exportar o modelo: {e}")
        raise


if __name__ == "__main__":
    treinar_modelo_precificacao()
