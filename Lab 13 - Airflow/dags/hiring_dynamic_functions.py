import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib
from glob import glob

# 1. Función para crear carpetas
def create_folders(execution_date):
    base_folder = os.path.join(os.getcwd(), f"output_{execution_date}")
    subfolders = ["raw", "splits", "models","preprocessed"]
    os.makedirs(base_folder, exist_ok=True)
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_folder, subfolder), exist_ok=True)
        print(f"Creating folders for {os.path.join(base_folder, subfolder)}")

import os
import pandas as pd

def load_and_merge(execution_date):
    """
    Lee los archivos `data_1.csv` y `data_2.csv` desde la carpeta `raw`,
    los concatena si ambos existen, y guarda el resultado en `preprocessed`.
    """
    # Definir rutas de las carpetas
    raw_folder = f"output_{execution_date}/raw"
    preprocessed_folder = f"output_{execution_date}/preprocessed"


    # Definir los archivos a buscar
    data_files = [os.path.join(raw_folder, f"data_{i}.csv") for i in range(1, 3)]

    # Leer los archivos que existen
    data_frames = []
    for file in data_files:
        if os.path.exists(file):
            print(f"Leyendo archivo: {file}")
            data_frames.append(pd.read_csv(file))
        else:
            print(f"Archivo no encontrado: {file}")

    if not data_frames:
        raise FileNotFoundError("No se encontraron archivos `data_1.csv` o `data_2.csv` en la carpeta `raw`.")

    # Concatenar DataFrames
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Guardar los datos combinados
    output_file = os.path.join(preprocessed_folder, "combined_data.csv")
    combined_data.to_csv(output_file, index=False)
    print(f"Datos combinados guardados en: {output_file}")


# 3. Dividir datos
def split_data(execution_date):
    """
    Divide el archivo `combined_data.csv` en conjuntos de entrenamiento y prueba (80%-20%).
    Guarda los conjuntos en la carpeta `splits`.
    """
    preprocessed_folder = f"output_{execution_date}/preprocessed"
    splits_folder = f"output_{execution_date}/splits"

    # Leer datos preprocesados
    data_path = os.path.join(preprocessed_folder, "combined_data.csv")
    data = pd.read_csv(data_path)

    # Dividir datos
    X = data.drop(columns=["HiringDecision"])
    y = data["HiringDecision"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Guardar datos
    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(splits_folder, "train.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(splits_folder, "test.csv"), index=False)

# 4. Entrenar modelo
def train_model(model, execution_date):
    """
    Entrena un modelo de clasificación utilizando los datos de entrenamiento.
    Guarda el modelo entrenado en la carpeta `models`.
    """
    splits_folder = f"output_{execution_date}/splits"
    models_folder = f"output_{execution_date}/models"

    # Leer conjunto de entrenamiento
    train_path = os.path.join(splits_folder, "train.csv")
    train_data = pd.read_csv(train_path)
    X_train = train_data.drop(columns=["HiringDecision"])
    y_train = train_data["HiringDecision"]

    # Configurar preprocesamiento
    numeric_features = ["Age", "ExperienceYears", "DistanceFromCompany", "InterviewScore", "SkillScore", "PersonalityScore"]
    categorical_features = ["Gender", "EducationLevel", "PreviousCompanies", "RecruitmentStrategy"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Crear pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    # Entrenar modelo
    pipeline.fit(X_train, y_train)

    # Guardar modelo entrenado
    model_name = type(model).__name__
    joblib.dump(pipeline, os.path.join(models_folder, f"{model_name}.joblib"))

# 5. Evaluar modelos
def evaluate_models(execution_date):
    """
    Evalúa los modelos entrenados y selecciona el mejor basado en `accuracy`.
    Guarda el mejor modelo como `best_model.joblib` en la carpeta `models`.
    """
    splits_folder = f"output_{execution_date}/splits"
    models_folder = f"output_{execution_date}/models"

    # Leer conjunto de prueba
    test_path = os.path.join(splits_folder, "test.csv")
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(columns=["HiringDecision"])
    y_test = test_data["HiringDecision"]

    # Evaluar modelos
    model_files = glob(os.path.join(models_folder, "*.joblib"))
    if not model_files:
        raise FileNotFoundError("No se encontraron modelos en la carpeta `models`.")

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for model_file in model_files:
        model = joblib.load(model_file)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        model_name = os.path.basename(model_file).replace(".joblib", "")
        print(f"Modelo: {model_name}, Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = model_name

    # Guardar mejor modelo
    if best_model:
        joblib.dump(best_model, os.path.join(models_folder, "best_model.joblib"))
        print(f"Mejor modelo: {best_model_name}, Accuracy: {best_accuracy:.4f}")
    else:
        raise ValueError("No se pudo determinar el mejor modelo.")

