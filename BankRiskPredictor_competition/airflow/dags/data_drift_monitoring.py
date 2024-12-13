from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.data.load_data import download_data
from src.data.preprocess_data import concatenate_datasets
from src.models.detect_drift import detect_data_drift
from src.models.optimize_hyperparameters import optimize_hyperparameters
from src.models.train_model import train_and_register_model

# Configuración de la DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "drift_detection_pipeline",
    default_args=default_args,
    description="Pipeline para detectar drift, optimizar y reentrenar el modelo",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Descargar datos
download_task = PythonOperator(
    task_id="download_data",
    python_callable=download_data,
    op_kwargs={"week": 2},
    dag=dag,
)

# Procesar datos
process_data_task = PythonOperator(
    task_id="process_data",
    python_callable=concatenate_datasets,
    op_kwargs={"weeks": [0, 1, 2]},
    dag=dag,
)

# Detectar drift
detect_drift_task = PythonOperator(
    task_id="detect_drift",
    python_callable=detect_data_drift,
    op_kwargs={
        "X_previous": "data/raw/X_t1.parquet",
        "X_current": "data/raw/X_t2.parquet",
    },
    dag=dag,
)

# Optimizar hiperparámetros
optimize_hyperparams_task = PythonOperator(
    task_id="optimize_hyperparameters",
    python_callable=optimize_hyperparameters,
    op_kwargs={
        "X": "data/processed/X_train.parquet",
        "y": "data/processed/y_train.parquet",
    },
    dag=dag,
)

# Evaluar modelo
evaluate_model_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    op_kwargs={
        "model": "models/rf_updated.pkl",
        "X_test": "data/processed/X_test.parquet",
        "y_test": "data/processed/y_test.parquet",
    },
    dag=dag,
)

# Interpretar modelo
interpret_model_task = PythonOperator(
    task_id="interpret_model",
    python_callable=interpret_model,
    op_kwargs={
        "model": "models/rf_updated.pkl",
        "X_sample": "data/processed/X_sample.parquet",
        "feature_names": "data/processed/feature_names.json",
    },
    dag=dag,
)


# Dependencias
download_task >> process_data_task >> detect_drift_task
detect_drift_task >> [optimize_hyperparams_task, retrain_model_task]  >> evaluate_model_task >> interpret_model_task

