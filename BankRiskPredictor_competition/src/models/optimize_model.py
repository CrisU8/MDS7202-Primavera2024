import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

def optimize_hyperparameters(X, y):
    """
    Optimizar los hiperparámetros del modelo utilizando Optuna.
    """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 5, 20)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        auc_pr = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        return auc_pr

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    return study.best_params
