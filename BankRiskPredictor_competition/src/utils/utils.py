import requests

def download_data(url: str, output_path: str):
    """
    Descarga un archivo desde una URL y lo guarda en un destino local.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica si hubo algún error en la solicitud
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"Archivo descargado correctamente: {output_path}")
    except Exception as e:
        print(f"Error al descargar el archivo: {e}")
        raise
