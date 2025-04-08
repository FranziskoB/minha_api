from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

with open("churn_model.pkl", "rb") as f:
    modelo = pickle.load(f)

class Entrada(BaseModel):
    features: list[float]

@app.post("/predict")
def prever(dados: Entrada):
    X = np.array(dados.features).reshape(1, -1)
    predicao = modelo.predict(X).tolist()[0]
    return {"features": dados.features, "predicao": predicao}
