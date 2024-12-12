FROM python:3.8-slim

WORKDIR /BankRiskPredictor_competition/frontend

# Instalar dependencias
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copiar el código de la app
COPY frontend.py

# Exponer el puerto de Gradio
EXPOSE 7860

# Ejecutar la app
CMD ["python", "frontend.py"]
