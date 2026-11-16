# 📈 S&P 500 Market Trend Prediction with PyTorch LSTM & MLOps

Pipeline end-to-end de **Deep Learning y MLOps** diseñado para predecir tendencias de mercado (clasificación binaria: alza/baja) utilizando Redes Neuronales Recurrentes (**LSTM**) implementadas con **PyTorch**, versionado de artefactos con **MLflow** y despliegue en tiempo real mediante una API de alta concurrencia en **FastAPI**.

---

## 🛠️ Arquitectura del Proyecto

El proyecto sigue una estructura modular orientada a producción:

```text
sp500-lstm-deep-learning-pipeline/
│
├── .github/
│   └── workflows/
│       └── ci_pipeline.yml     # Automatización CI/CD con GitHub Actions
├── data/
│   ├── raw/                    # Datos crudos del S&P 500 (.parquet)
│   └── processed/              # Datos limpios y estructurados
├── mlruns/                     # Registro local de experimentos y artefactos de MLflow
├── src/
│   ├── api/
│   │   └── predict_api.py      # Servicio web de inferencia en tiempo real (FastAPI)
│   ├── data/
│   │   └── features.py         # Ingeniería de características y creación de secuencias temporales
│   └── models/
│       └── train_lstm.py       # Arquitectura PyTorch LSTM, entrenamiento y tracking MLflow
├── requirements.txt            # Dependencias del proyecto
└── README.md


Guía de Instalación y Ejecución Local
1. Clonar el repositorio e instalar dependencias

    git clone [https://github.com/TU_USUARIO/sp500-lstm-deep-learning-pipeline.git](https://github.com/TU_USUARIO/sp500-lstm-deep-learning-pipeline.git)
    cd sp500-lstm-deep-learning-pipeline
    pip install -r requirements.txt

2. Generar características y entrenar el modelo

    # Procesar datos y generar secuencias
    python src/data/features.py

    # Entrenar la red LSTM y registrar en MLflow
    python src/models/train_lstm.py


3. Levantar la API de Inferencia

    python -m uvicorn src.api.predict_api:app --reload --port 8000