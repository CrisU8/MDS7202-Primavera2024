import optuna
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def optimize_model():
    # Cargar datos
    data = pd.read_csv("water_potability.csv")
    data.dropna(inplace=True)

    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Crear carpetas para artefactos
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Función objetivo para Optuna
    def objective(trial):
        with mlflow.start_run(run_name=f"XGBoost_Trial_{trial.number}"):
            # Sugerir hiperparámetros
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            }

            # Entrenar modelo
            model = xgb.XGBClassifier(**params, use_label_encoder=False)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

            # Calcular f1-score
            y_pred = model.predict(X_valid)
            valid_f1 = f1_score(y_valid, y_pred)

            # Registrar métricas e hiperparámetros en MLflow
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", valid_f1)

            # Guardar gráfica de importancia de características
            plt.figure(figsize=(10, 5))
            xgb.plot_importance(model, max_num_features=10)
            plt.savefig("plots/feature_importance.png")
            mlflow.log_artifact("plots/feature_importance.png", artifact_path="plots")
            plt.close()

            return valid_f1

    # Configurar MLflow
    mlflow.set_experiment("Optimización XGBoost Agua Potable")

    # Crear estudio Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Mejor hiperparámetro
    best_params = study.best_params
    print("Mejores Hiperparámetros:", best_params)

    # Entrenar el mejor modelo
    best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
    best_model.fit(X_train, y_train)

    # Guardar el modelo final
    with open("models/best_xgboost_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Guardar gráfica final de importancia de características
    plt.figure(figsize=(10, 5))
    xgb.plot_importance(best_model, max_num_features=10)
    plt.savefig("plots/final_feature_importance.png")
    plt.close()

    print("Modelo guardado en: models/best_xgboost_model.pkl")

if __name__ == "__main__":
    optimize_model()