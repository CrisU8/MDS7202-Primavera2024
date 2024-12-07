import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import joblib
import gradio as gr


# 1. Función para crear carpetas
def create_folders(execution_date):
    base_folder = os.path.join(os.getcwd(), f"output_{execution_date}")
    subfolders = ["raw", "splits", "models"]
    os.makedirs(base_folder, exist_ok=True)
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_folder, subfolder), exist_ok=True)
        print(f"Creating folders for {os.path.join(base_folder, subfolder)}")



# 2. Función para dividir los datos
def split_data(execution_date):
    # Define rutas basadas en la fecha de ejecución
    base_path = os.getcwd()
    input_path = os.path.join(base_path, f"output_{execution_date}/raw/data_1.csv")
    split_folder = os.path.join(base_path, f"output_{execution_date}/splits")

    # Verifica si el archivo de entrada existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    # Lee el archivo y divide los datos
    data = pd.read_csv(input_path)
    if "HiringDecision" not in data.columns:
        raise ValueError("La columna 'HiringDecision' no está en los datos.")

    X = data.drop(columns=["HiringDecision"])
    y = data["HiringDecision"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Guarda los conjuntos divididos
    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(split_folder, "train.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(split_folder, "test.csv"), index=False)
    print(f"Archivos guardados en {split_folder}")



# 3. Función para preprocesar y entrenar modelo
def preprocess_and_train(execution_date):
    # Leer datasets
    base_path = os.getcwd()
    train_path = os.path.join(base_path, f"output_{execution_date}/splits/train.csv")
    test_path = os.path.join(base_path, f"output_{execution_date}/splits/test.csv")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=["HiringDecision"])
    y_train = train_data["HiringDecision"]
    X_test = test_data.drop(columns=["HiringDecision"])
    y_test = test_data["HiringDecision"]

    # Preprocesamiento
    numeric_features = ["Age", "ExperienceYears", "DistanceFromCompany", "InterviewScore", "SkillScore",
                        "PersonalityScore"]
    categorical_features = ["Gender", "EducationLevel", "PreviousCompanies", "RecruitmentStrategy"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Pipeline de modelo
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Entrenar modelo
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Positive Class): {f1:.4f}")

    # Guardar modelo entrenado
    model_path = f"output_{execution_date}/models/hiring_model.joblib"
    joblib.dump(model, model_path)


def predict(file,model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface(execution_date):

    model_path= f'output_{execution_date}/models/hiring_model.joblib'

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Nico será contratado o no."
    )
    interface.launch(share=True)
