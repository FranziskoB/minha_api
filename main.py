from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pode restringir para domínios específicos em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelo
with open("churn_model.pkl", "rb") as f:
    modelo = pickle.load(f)

class Entrada(BaseModel):
    features: list[float]

@app.post("/predict")
def prever(dados: Entrada):
    X = np.array(dados.features).reshape(1, -1)
    if hasattr(modelo, "predict_proba"):
        prob_churn = modelo.predict_proba(X)[0][1]
        return {"probabilidade_churn_percentual": round(prob_churn * 100, 2)}
    else:
        return {"erro": "Modelo não suporta predict_proba"}
