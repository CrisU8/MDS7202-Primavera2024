import pandas as pd

def concatenate_datasets(weeks):
    """
    Concatenar datasets históricos para generar el dataset completo.
    """
    X = pd.concat([pd.read_parquet(f"data/raw/X_t{i}.parquet") for i in weeks], axis=0)
    y = pd.concat([pd.read_parquet(f"data/raw/y_t{i}.parquet") for i in weeks], axis=0)
    return X, y

def preprocess_pipeline():
    