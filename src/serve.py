from fastapi import FastAPI
import joblib
import pandas as pd
import uvicorn

# Carregar modelo treinado
model = joblib.load("models/model.pkl")

app = FastAPI(title="Energy Demand Prediction API")

@app.post("/predict")
def predict(features: dict):
    """
    Exemplo de payload esperado:
    {
        "temperatura": 28,
        "hora_dia": 15,
        "dia_semana": 2
    }
    """
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return {"predicted_consumo": float(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
