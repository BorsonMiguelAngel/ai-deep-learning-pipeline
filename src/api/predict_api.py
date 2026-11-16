import os 
from pathlib import Path
import sys
import mlflow
import mlflow.pytorch
import numpy as np
from fastapi import FastAPI, HTTPException
import pydantic
import torch
import torch.nn as nn

# Obtenemos la ruta absoluta de la raiz del proyecto. Subimos dos niveles desde src/models
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Importamos nuestra funcion del otro archivo (demostrando la modularidad)
from src.models.train_lstm import S500LSTMClassifier

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="SP500 Deep Learning MLOps API",
    description=(
        "API de inferencia en tiempo real para predicción de series temporales"
        " con PyTorch LSTM y MLflow."
    ),
    version="2.0",
)

# Variable global para cachear el modelo en memoria y evitar recargarlo en cada petición
model = None

def load_latest_mlflow_model():
  """Busca y carga automáticamente el modelo LSTM más reciente registrado en MLflow."""
  global model
  if model is not None:
    return model

  try:
    # Apuntar al directorio local de MLflow
    mlflow_dir = ROOT_DIR / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlflow_dir.resolve()}")

    # Buscar experimentos y runs locales para encontrar el artefacto del modelo
    # Ruta estándar de MLflow para artefactos guardados
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    if not experiments:
      raise RuntimeError("No se encontraron experimentos en MLflow.")

    # Tomar el experimento más reciente
    exp_id = experiments[0].experiment_id
    runs = client.search_runs(
        experiment_ids=[exp_id], order_by=["start_time DESC"], max_results=1
    )

    if not runs:
      raise RuntimeError("No se encontraron ejecuciones (runs) en MLflow.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/lstm_model"

    print(f"Cargando modelo LSTM desde MLflow (Run ID: {run_id})...")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()  # Modo evaluación (desactiva dropout/batchnorm si los hubiera)
    return model

  except Exception as e:
    print(
        f"Advertencia: No se pudo cargar el modelo desde MLflow automáticamente:"
        f" {e}"
    )
    # Fallback o manejo de error si el modelo no está entrenado aún
    return None


# ===================================
# ESQUEMA DE ENTRADA CON PYDANTIC
# =======================================
class LSTMInputData(pydantic.BaseModel):
  # Espera una lista bidimensional: 30 días (time steps) x características por día
  # Ejemplo: 30 filas, cada una con [Return, MA_10, MA_30, Volatility]
  sequence: list[list[float]]


@app.get("/")
def home():
  return {
      "status": "Online",
      "model": "PyTorch LSTM S500 Classifier",
      "description": "Envía una secuencia de 30 días a /predict",
  }


@app.post("/predict")
def predict(data: LSTMInputData):
  global model
  if model is None:
    model = load_latest_mlflow_model()
    if model is None:
      raise HTTPException(
          status_code=500,
          detail=(
              "El modelo LSTM no está disponible. Entrena el modelo primero"
              " ejecutando train_lstm.py"
          ),
      )

  try:
    # 1. Convertir la entrada JSON a un arreglo de NumPy y luego a Tensor de PyTorch
    input_array = np.array(data.sequence, dtype=np.float32)

    # Validar que la forma tenga exactamente 30 pasos de tiempo (según definimos en el entrenamiento)
    # Esperado: (30, num_features) o con batch (1, 30, num_features)
    if len(input_array.shape) == 2:
      # Añadir dimensión de batch al inicio -> (1, 30, num_features)
      input_tensor = torch.tensor(input_array).unsqueeze(0)
    elif len(input_array.shape) == 3:
      input_tensor = torch.tensor(input_array)
    else:
      raise ValueError(
          "Formato de secuencia inválido. Se espera una matriz 2D o tensor 3D."
      )

    # 2. Ejecutar inferencia con PyTorch sin calcular gradientes
    with torch.no_grad():
      prediction_prob = model(input_tensor).item()

    # 3. Definir la decisión basada en la probabilidad de la sigmoide (umbral 0.5)
    prediction_class = 1 if prediction_prob >= 0.5 else 0
    signal = (
        "ALCISTA (Sube)" if prediction_class == 1 else "BAJISTA (Baja/Neutral)"
    )

    return {
        "prediction_class": prediction_class,
        "probability": round(prediction_prob, 4),
        "market_signal": signal,
        "message": (
            "Predicción generada exitosamente mediante red neuronal LSTM."
        ),
    }

  except Exception as e:
    raise HTTPException(
        status_code=400, detail=f"Error al procesar la inferencia: {str(e)}"
    )