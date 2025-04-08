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
    predicao = modelo.predict(X)
    return {"predicao": predicao.tolist()}
