import pandas as pd

def download_data(week):
    """
    Descarga los datos para una semana específica y los guarda localmente.
    """
    base_url = "https://gitlab.com/mds7202-2/proyecto-mds7202/-/blob/main/competition_files/"
    files = [f"X_t{week}.parquet", f"y_t{week}.parquet"]
    local_paths = []
    for file in files:
        url = f"{base_url}{file}?ref_type=heads"
        local_path = f"data/raw/{file}"
        pd.read_parquet(url).to_parquet(local_path)
        local_paths.append(local_path)
    return local_paths
