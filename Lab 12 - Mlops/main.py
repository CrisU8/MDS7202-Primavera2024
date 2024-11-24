from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Cargar el modelo previamente optimizado y guardado
with open("models/best_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Inicializar la aplicación FastAPI
app = FastAPI()

# Definir el esquema de datos para la entrada del endpoint
class WaterQuality(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Ruta POST para predecir la potabilidad
@app.post("/potabilidad/")
def predict_water_quality(data: WaterQuality):
    # Convertir los datos del request a un array para el modelo
    input_data = np.array([
        data.ph,
        data.Hardness,
        data.Solids,
        data.Chloramines,
        data.Sulfate,
        data.Conductivity,
        data.Organic_carbon,
        data.Trihalomethanes,
        data.Turbidity
    ]).reshape(1, -1)

    # Hacer la predicción con el modelo optimizado
    prediction = model.predict(input_data)[0]

    # Retornar la respuesta en formato JSON
    return {"potabilidad": int(prediction)}

# Código para levantar el servidor con python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
