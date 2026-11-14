import os 
from pathlib import Path
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Importamos nuestra funcion del otro archivo (demostrando la modularidad)
from src.data.features import create_sentences

# Configuracion obligatoria para permitir almacenamiento local en MLflow (evita el error de mantenimiento)
os.environ("MLFLOW_ALLOW_FILE_STORE") = "true"

"""
    1.- DEFINICION DE LA ARQUITECTURA DE LA RED LSTM
"""
class S500LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=1):
        super(S500LSTMClassifier, self).__init__()
        # Capa LSTM: Procesa la secuencia temporal
        # batch_first=True indica que la forma de la entrada es (Batch, Time_Steps, features)
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim, 
            num_layers = num_layers,
            batch_first = True,
        )
        # Capa lineal (Denso): Convierte la salida oculta de LSTM en una probabilidad de clasificacion 
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Capa Sigmoide: Restringe la salida entre un valor de 0 a 1 (Para clasificacion binaria: sube o baja) 
        self.sigmoid = nn.Sigmoid()

    def foward(self, x):
        # Forma de x: (batch_size, seq_len, input_dim)
        # Forma de recurrent_out: (batch_size, seq_len, hidden_dim)
        # _ (hidden state y cell state) se descartan en este enfoque simple
        recurrent_out, _ = self.lstm(x)

        # Tomamos unicamente la salida correspondiente al ultimo paso de tiempo de la secuencia
        last_time_step = recurrent_out[:, -1, :]

        # Pasamos por la capa lineal y aplicamos la sigmoide
        out = self.fc(last_time_step)
        return self.sigmoid(out) 

# ======================================
# 2. FUNCION PRINCIPAL DE ENTRENAMIENTO Y MLFLOW
# ==================================================

def train_lstm_model(
    data_path="data/processed/sp500_caracteristicas.parquet",
    window_size=30,
    epochs=10,
    batch_size=32,
    lr=0.001,
):
    print("---Iniciando pipeline de entrenamiento con Pytorch y LSTM---")

    # 1. Configurar MLflow
    mlflow.set_tracking_uri(f"file://{Path('mlruns').resolve()}")
    mlflow.set_experiment("SP500_DeepLearning_LSTM")

    # 2. Cargar datos procesados
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"No se encontro el archivo procesado en {data_path}. Ejecuta features.py primero"
    )

    df = pd.read_parquet(data_path)
    target_col = "Target"
    
    print(
        f"Generando secuencias temporales con una ventana de {window_size} días..."
    )

    # Aqui se invoca la funcion del archivo "features.py"
    X, y = create_sentences(df, target_col=target_col, windows_size=window_size)

    print(f"Forma de las caracteristicas X (Tensores): {X.shape}")
    print(f"Forma de la etiqueta y: {y.shape}")

    # 3. Division Cronologica (Train / Test split sin mezclar el tiempo)
    # En series de tiempo NUNCA se usa train_test_split aleatorio para evitar fuga de datos (data leakage)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 4. Convertir arreglos de NumPy a Tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Dimensión (N, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Crear DataLoaders para manejar lotes (batches) eficientemente
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )  # Shuffle=False respeta el orden temporal en series de tiempo

    # 5. Inicializar Modelo, Función de Pérdida y Optimizador
    input_dim = X.shape[2]  # Número de características predictoras por cada día
    model = S500LSTMClassifier(input_dim=input_dim, hidden_dim=32, output_dim=1)

    criterion = (
        nn.BCELoss()
    )  # Binary Cross Entropy para clasificación de 0 o 1 (sube/baja)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Iniciar corrida en MLflow
    with mlflow.start_run() as run:
        print(f"Iniciando corrida en MLflow (Run ID: {run.info.run_id})...")

        # Registrar hiperparámetros clave
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)

        # 7. Ciclo de Entrenamiento (Epochs)
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()  # Limpiar gradientes anteriores
                predictions = model(batch_X)  # Propagación hacia adelante (Forward)
                loss = criterion(predictions, batch_y)  # Calcular el error
                loss.backward()  # Retropropagación (Backward)
                optimizer.step()  # Actualizar pesos sinápticos

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(
                f"Época [{epoch+1}/{epochs}] - Pérdida (Loss Training): {avg_loss:.4f}"
            )
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # 8. Evaluación rápida en el conjunto de prueba
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_tensor)
            test_preds_binary = (test_preds >= 0.5).float()
            accuracy = (test_preds_binary == y_test_tensor).float().mean().item()

        print(
            f"Exactitud (Accuracy) del modelo LSTM en Test:"
            f" {accuracy * 100:.2f}%"
        )
        mlflow.log_metric("test_accuracy", accuracy)

        # 9. Registrar el modelo entrenado de PyTorch en MLflow
        mlflow.pytorch.log_model(model, artifact_path="lstm_model")
        print(
            "Modelo PyTorch LSTM entrenado y registrado exitosamente en MLflow"
            " artifacts."
        )


    if __name__ == "__main__":
        train_lstm_model()