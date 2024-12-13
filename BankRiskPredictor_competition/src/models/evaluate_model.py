import mlflow
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba y registra las métricas en MLflow.
    """
    mlflow.set_experiment("drift_detection_pipeline")
    with mlflow.start_run():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calcular métricas
        auc_score = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Registrar métricas en MLflow
        mlflow.log_metric("AUC", auc_score)
        mlflow.log_metrics({
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"]
        })

        # Generar y registrar curvas de Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig("precision_recall_curve.png")
        plt.close()

        mlflow.log_artifact("precision_recall_curve.png")

        return auc_score, report
