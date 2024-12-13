def detect_data_drift(X_previous, X_current):
    """
    Detectar drift en los datos comparando estadísticas básicas.
    """
    drift_detected = not X_previous.mean().equals(X_current.mean())
    return drift_detected
