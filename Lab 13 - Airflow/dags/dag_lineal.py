from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface


# Inicialización del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Definición del DAG
with DAG(
    dag_id="hiring_lineal",
    default_args=default_args,
    description="Pipeline para predicción de contratación",
    schedule_interval=None,  # Ejecución manual
    start_date=datetime(2024, 10, 1),
    catchup=False,  # Sin backfill
    params={}
) as dag:

    # 1. Marcador de inicio
    start = DummyOperator(task_id="start_pipeline")

    # 2. Crear carpetas
    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    # 3. Descargar datos
    download_data_task = BashOperator(
        task_id='download_dataset',
        bash_command=(
            "curl -o /root/airflow/output_{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ),
    )

    # 4. Aplicar hold out
    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        provide_context=True,
        op_kwargs={"execution_date": "{{ ds }}"}
    )

    # 5. Preprocesar y entrenar
    preprocess_and_train_task = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        provide_context=True,
        op_kwargs={"execution_date": "{{ ds }}"}
    )

    # 6. Generar interfaz con Gradio
    gradio_interface_task = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
        provide_context=True,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    # 7. Finalizar
    end = DummyOperator(task_id="end_pipeline")

    # Definición del flujo de tareas
    start >> create_folders_task >> download_data_task >> split_data_task >> preprocess_and_train_task >> gradio_interface_task >> end