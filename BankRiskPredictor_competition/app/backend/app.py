from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional, Dict, Any
import numpy as np

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
# Custom Transformer para procesar timestamps y crear 'wallet_age_hours'
class WalletAgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['last_tx_timestamp'] = pd.to_datetime(X['last_tx_timestamp'])
        X['first_tx_timestamp'] = pd.to_datetime(X['first_tx_timestamp'])
        X['wallet_age_hours'] = (X['last_tx_timestamp'] - X['first_tx_timestamp']).dt.total_seconds() / 360
        return X

# Custom Transformer para agregar clusters
class ClusterAdder(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=4, random_state=42):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        clusters = self.kmeans.predict(X)
        return np.hstack((X, clusters.reshape(-1, 1)))

# Crear la app FastAPI
app = FastAPI()

# Cargar el modelo y el preprocesador
try:
    preprocessor = joblib.load("../preprocessor.pkl")
    model = joblib.load("../rf_updated.pkl")
except Exception as e:
    raise RuntimeError(f"Error al cargar modelo o preprocesador: {str(e)}")

class PredictionInput(BaseModel):
    last_tx_timestamp: int  # Los timestamps se envían como strings en formato ISO 8601
    first_tx_timestamp: int
    wallet_age_hours: float
    risk_factor: float
    avg_risk_factor: float
    time_since_last_liquidated: float
    market_aroonosc: float
    unique_borrow_protocol_count: int
    risk_factor_above_threshold_daily_count: int
    outgoing_tx_avg_eth: float
    max_eth_ever: float
    total_gas_paid_eth: float
    avg_gas_paid_per_tx_eth: float
    net_incoming_tx_count: int

# Ruta para predicciones por archivo .csv
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile):
    try:
        # Leer el archivo CSV subido
        df = pd.read_csv(file.file)

        # Aplicar el preprocesador a los datos
        X = preprocessor.transform(df)

        # Realizar las predicciones y calcular las probabilidades
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Combinar los resultados en una lista de diccionarios
        results = [
            {"prediction": int(pred), "probability": float(prob)}
            for pred, prob in zip(predictions, probabilities)
        ]

        return {"predictions": results}

    except Exception as e:
        # Manejar cualquier error que ocurra en el flujo
        return {"error": f"Error durante la predicción: {str(e)}"}



# Ruta para predicciones manuales
@app.post("/predict_manual/")
async def predict_manual(input_data: PredictionInput):
    try:
        # Convertir el modelo de entrada a un DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])

        # Aplicar el preprocesador
        X = preprocessor.transform(df)

        # Realizar la predicción
        prediction = int(model.predict(X)[0])  # Convertir a entero
        probability = float(model.predict_proba(X)[:, 1][0])  # Convertir a flotante

        return {"prediction": prediction, "probability": probability}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "error": f"Error durante la predicción: {str(e)}",
            "details": error_details
        }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)