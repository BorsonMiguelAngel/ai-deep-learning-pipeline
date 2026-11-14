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