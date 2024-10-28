from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import subprocess

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 21),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fraud_data_generation_dag',
    default_args=default_args,
    schedule_interval=timedelta(days=1),  # Run daily
)

def generate_fraud_data():
    subprocess.run(["python", r"Data_Set\data.py"])

generate_task = PythonOperator(
    task_id='generate_fraud_data',
    python_callable=generate_fraud_data,
    dag=dag,
)

generate_task