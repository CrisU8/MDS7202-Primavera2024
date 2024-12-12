FROM python:3.8-slim

WORKDIR /BankRiskPredictor_competition/backend

# Instalar dependencias
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copiar el código de la app
COPY app.py

# Exponer el puerto de FastAPI
EXPOSE 8000

# Ejecutar la app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
