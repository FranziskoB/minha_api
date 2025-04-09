from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Criação do app FastAPI
app = FastAPI()

# Carregar o modelo treinado
with open("churn_model.pkl", "rb") as f:
    modelo = pickle.load(f)

# Modelo dos dados de entrada
class Entrada(BaseModel):
    features: list[float]  # ou dict, depende do modelo

@app.post("/predict")
def prever(dados: Entrada):
    X = np.array(dados.features).reshape(1, -1)
    
    # Verifica se o modelo possui o método predict_proba
    if hasattr(modelo, "predict_proba"):
        probabilidade_churn = modelo.predict_proba(X)[0][1]  # Probabilidade da classe positiva (churn)
        percentual = round(probabilidade_churn * 100, 2)  # Converte para percentual com 2 casas decimais
        return {"probabilidade_churn_percentual": percentual}
    else:
        return {"erro": "O modelo não suporta previsão de probabilidade (predict_proba)."}
