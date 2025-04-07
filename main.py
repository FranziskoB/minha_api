from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"mensagem": "API FastAPI no ar com Render!"}

@app.get("/soma")
def somar(a: int, b: int):
    return {"resultado": a + b}