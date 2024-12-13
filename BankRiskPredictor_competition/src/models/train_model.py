import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow

def train_and_register_model(X, y, params):
    """
    Entrenar un modelo con los datos dados y registrar en MLflow.
    """
    mlflow.set_experiment("drift_detection_pipeline")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        # Registrar en MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Guardar el modelo localmente
        joblib.dump(model, "models/rf_updated.pkl")

        # Registrar el modelo en MLflow
        mlflow.sklearn.log_model(model, "model")
