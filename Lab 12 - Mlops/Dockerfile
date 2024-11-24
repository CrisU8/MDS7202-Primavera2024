# Usar imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /Lab 12 - Mlops

# Copiar requirements.txt y los archivos del proyecto
COPY requirements.txt ./
COPY main.py ./
COPY models ./models

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8000
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


