from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
from hiring_dynamic_functions import create_folders, load_and_merge, split_data, train_model, evaluate_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
from datetime import datetime, timezone


from datetime import datetime, timezone

def decide_which_datasets(**kwargs):
    """
    Decide qué datasets descargar basado en la fecha de ejecución.
    """
    # Obtener execution_date del contexto
    execution_date = kwargs['ds']

    # Crear fecha límite
    cutoff_date = '2024-11-01'

    # Validar y comparar las fechas
    if execution_date < cutoff_date:
        print(f"Execution date {execution_date} es anterior a {cutoff_date}. Descargando data_1.csv.")
        return "download_data_1"
    else:
        print(f"Execution date {execution_date} es igual o posterior a {cutoff_date}. Descargando data_1.csv y data_2.csv.")
        return "download_data_1_and_2"


# Configuración del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id="hiring_dynamic_pipeline",
    default_args=default_args,
    description="Pipeline dinámico para entrenamiento de modelos de contratación",
    schedule_interval="0 15 5 * *",  # Ejecutar el día 5 de cada mes a las 15:00 UTC
    start_date=datetime(2024, 10, 1),
    catchup=True,  # Habilitar backfill
) as dag:

    # Marcador de inicio
    start = DummyOperator(task_id="start_pipeline")

    # Crear carpetas
    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    branch_task = BranchPythonOperator(
        task_id="decide_datasets",
        python_callable=decide_which_datasets,
        provide_context=True,  # Para acceder a execution_date
    )

    # Descargar data_1.csv
    download_data_1_task = BashOperator(
        task_id='download_data_1',
        bash_command=(
            "curl -o /root/airflow/output_{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ),
    )

    # Descargar data_1.csv y data_2.csv
    download_data_1_and_2_task = BashOperator(
        task_id='download_data_1_and_2',
        bash_command=(
             "curl -o /root/airflow/output_{{ ds }}/raw/data_1.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv && "
        "curl -o /root/airflow/output_{{ ds }}/raw/data_2.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
    ),
)

    # Concatenar datasets
    load_and_merge_task = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        op_kwargs={"execution_date": "{{ ds }}"},
        trigger_rule=TriggerRule.ONE_SUCCESS,  # Ejecutar si al menos una rama del Branch fue exitosa
    )

    # Aplicar hold-out
    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    # Entrenamiento de modelos
    train_rf_task = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_model,
        op_kwargs={"model": RandomForestClassifier(random_state=42), "execution_date": "{{ ds }}"},
    )
    train_logreg_task = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_model,
        op_kwargs={"model": LogisticRegression(max_iter=1000), "execution_date": "{{ ds }}"},
    )
    train_svc_task = PythonOperator(
        task_id="train_svc",
        python_callable=train_model,
        op_kwargs={"model": SVC(), "execution_date": "{{ ds }}"},
    )

    # Evaluar modelos
    evaluate_models_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={"execution_date": "{{ ds }}"},
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Ejecutar solo si todos los modelos fueron entrenados
    )

    # Marcador de fin
    end = DummyOperator(task_id="end_pipeline")

    # Flujo de tareas
    start >> create_folders_task >> branch_task
    branch_task >> [download_data_1_task, download_data_1_and_2_task]
    [download_data_1_task, download_data_1_and_2_task] >> load_and_merge_task >> split_data_task
    split_data_task >> [train_rf_task, train_logreg_task, train_svc_task] >> evaluate_models_task >> end
