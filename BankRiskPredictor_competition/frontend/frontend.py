import gradio as gr
import requests

# Ruta del backend
BACKEND_URL = "http://localhost:8000"

# Función para predicción por archivo
def predict_csv(file):
    files = {'file': file}
    response = requests.post(f"{BACKEND_URL}/predict_csv/", files=files)
    return response.json()

# Función para predicción manual
def predict_manual(*args):
    input_data = {
        "last_tx_timestamp": args[0],
        "first_tx_timestamp": args[1],
        "risk_factor": args[2],
        "avg_risk_factor": args[3],
        "time_since_last_liquidated": args[4],
        "market_aroonosc": args[5],
        "unique_borrow_protocol_count": args[6],
        "risk_factor_above_threshold_daily_count": args[7],
        "outgoing_tx_avg_eth": args[8],
        "max_eth_ever": args[9],
        "total_gas_paid_eth": args[10],
        "avg_gas_paid_per_tx_eth": args[11],
        "net_incoming_tx_count": args[12]
    }
    response = requests.post(f"{BACKEND_URL}/predict_manual/", json=input_data)
    return response.json()

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("### Predicción de Probabilidad de Morosidad")

    with gr.Tab("Cargar Archivo CSV"):
        file_input = gr.File(label="Sube tu archivo CSV")
        csv_output = gr.Textbox(label="Predicciones")
        file_btn = gr.Button("Predecir")
        file_btn.click(predict_csv, inputs=file_input, outputs=csv_output)

    with gr.Tab("Ingreso Manual"):
        inputs = [
            gr.Number(label="wallet_age_hours"),
            gr.Number(label="risk_factor"),
            gr.Number(label="avg_risk_factor"),
            gr.Number(label="time_since_last_liquidated"),
            gr.Number(label="market_aroonosc"),
            gr.Number(label="unique_borrow_protocol_count"),
            gr.Number(label="risk_factor_above_threshold_daily_count"),
            gr.Number(label="outgoing_tx_avg_eth"),
            gr.Number(label="max_eth_ever"),
            gr.Number(label="total_gas_paid_eth"),
            gr.Number(label="avg_gas_paid_per_tx_eth"),
            gr.Number(label="net_incoming_tx_count"),
        ]
        manual_output = gr.Textbox(label="Predicción y Probabilidad")
        manual_btn = gr.Button("Predecir")
        manual_btn.click(predict_manual, inputs=inputs, outputs=manual_output)

# Ejecutar la aplicación Gradio
demo.launch()
