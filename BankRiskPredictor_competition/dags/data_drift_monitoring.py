from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "data_drift_monitoring",
        default_args=default_args,
        description="Pipeline para monitorear y gestionar data drift",
        schedule_interval="@weekly",
        start_date=datetime(2023, 1, 1),
        catchup=False,
) as dag:
    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=fetch_new_files,
        op_kwargs={"directory_url": "URL", "local_path": "/data/raw"},
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=apply_preprocessing,
        op_kwargs={"data_path": "/data/raw", "output_path": "/data/processed"},
    )

    drift_task = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_data_drift,
        op_kwargs={"original_data_path": "/data/processed/X_t0.parquet",
                   "new_data_path": "/data/processed/latest.parquet"},
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model_if_drift_detected,
    )

    extract_task >> transform_task >> drift_task >> retrain_task
