import gradio as gr
import requests

# Ruta del backend
BACKEND_URL = "http://localhost:8000/"

# Función para predicción por archivo
def predict_csv(file_path):
    try:
        with open(file_path.name, 'rb') as f:
            response = requests.post(f"{BACKEND_URL}/predict_csv/", files={"file": f})

        print("Respuesta del servidor:", response.status_code, response.text)  # Depuración

        # Verificar el estado de la respuesta
        response.raise_for_status()

        # Intentar decodificar el JSON
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error al realizar la solicitud: {str(e)}"}
    except requests.exceptions.JSONDecodeError:
        return {"error": "El servidor no devolvió una respuesta JSON válida."}


import requests

def predict_manual(*args):
    # Número esperado de argumentos
    expected_args = 13

    # Verificar que se proporcionen los argumentos necesarios
    if len(args) < expected_args:
        return {"error": f"Se esperaban {expected_args} argumentos, pero se recibieron {len(args)}"}

    # Construir los datos de entrada para la API
    input_data = {
        "last_tx_timestamp": args[0],  # Asumiendo que es un timestamp en formato ISO 8601
        "first_tx_timestamp": args[1],
        "wallet_age_hours": args[2],
        "risk_factor": args[3],
        "avg_risk_factor": args[4],
        "time_since_last_liquidated": args[5],
        "market_aroonosc": args[6],
        "unique_borrow_protocol_count": args[7],
        "risk_factor_above_threshold_daily_count": args[8],
        "outgoing_tx_avg_eth": args[9],
        "max_eth_ever": args[10],
        "total_gas_paid_eth": args[11],
        "avg_gas_paid_per_tx_eth": args[12],
        "net_incoming_tx_count": args[13],
    }

    # Realizar la solicitud POST a la API
    try:
        response = requests.post(f"{BACKEND_URL}/predict_manual/", json=input_data)
        response.raise_for_status()  # Genera una excepción si el código de estado HTTP indica un error
        return response.json()  # Retorna el JSON de la respuesta
    except requests.exceptions.RequestException as e:
        return {"error": f"Error al realizar la solicitud: {str(e)}"}



# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("### Predicción de Probabilidad de Morosidad")

    # Tab para cargar archivo CSV
    with gr.Tab("Cargar Archivo CSV"):
        file_input = gr.File(label="Sube tu archivo CSV")
        csv_output = gr.Textbox(label="Resultados del CSV")
        file_btn = gr.Button("Predecir desde CSV")
        file_btn.click(predict_csv, inputs=file_input, outputs=csv_output)

    # Tab para ingreso manual
    with gr.Tab("Ingreso Manual"):
        inputs = [
            gr.Textbox(label="Last Transaction Timestamp", placeholder="2023-01-01T12:34:56"),
            gr.Textbox(label="First Transaction Timestamp", placeholder="2023-01-01T10:30:00"),
            gr.Number(label="Wallet Age (Horas)"),
            gr.Number(label="Risk Factor"),
            gr.Number(label="Average Risk Factor"),
            gr.Number(label="Time Since Last Liquidated"),
            gr.Number(label="Market Aroonosc"),
            gr.Number(label="Unique Borrow Protocol Count"),
            gr.Number(label="Risk Factor Above Threshold Daily Count"),
            gr.Number(label="Outgoing TX Average ETH"),
            gr.Number(label="Max ETH Ever"),
            gr.Number(label="Total Gas Paid ETH"),
            gr.Number(label="Average Gas Paid Per TX ETH"),
            gr.Number(label="Net Incoming TX Count"),
        ]
        manual_output = gr.Textbox(label="Predicción y Probabilidad")
        manual_btn = gr.Button("Predecir")
        manual_btn.click(predict_manual, inputs=inputs, outputs=manual_output)


# Ejecutar la aplicación Gradio
demo.launch()
