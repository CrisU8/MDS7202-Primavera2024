import shap
import matplotlib.pyplot as plt
import mlflow

def interpret_model(model, X_sample, feature_names):
    """
    Genera interpretaciones del modelo utilizando SHAP y las registra en MLflow.
    """
    mlflow.set_experiment("drift_detection_pipeline")
    with mlflow.start_run():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Crear resumen gráfico
        shap.summary_plot(shap_values[1], X_sample, feature_names=feature_names, show=False)
        plt.savefig("shap_summary_plot.png")
        plt.close()

        # Registrar el gráfico en MLflow
        mlflow.log_artifact("shap_summary_plot.png")

        return shap_values
