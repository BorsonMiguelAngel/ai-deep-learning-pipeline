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