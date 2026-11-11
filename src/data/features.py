from pathlib import Path
import pandas as pd
import numpy as np

def create_sentences(data: pd.DataFrame, target_col: str, windows_size: int = 30):
    """ Transforma un DataFrame tabular en secuencias temporales (X)
    y su respectivo objetivo (y) para redes LSTM
    """
    features = data.drop(columns=[target_col, "Date"], errors="ignore").values
    target = data[target_col].values

    X, y = [], []
    for i in range(windows_size, len(data)):
        X.append(features[i - windows_size : i])
        y.append(target[i])

    return np.array(X), np.array(y)

def crear_caracteristicas_financieras(input_path: str):
    print(f"###====>XXX Cargando datos crudos desde '{input_path}'...")

    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de datos crudos en: {input_path}"
        )

    df = pd.read_parquet(input_path)

    #1. Aseguramos que los datos esten en orden cronologico por seguridad
    df = df.sort_values("Date").reset_index(drop=True)

    #2. Ingenieria de caracteristicas basicas (Retornos y medias moviles)
    df["Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_30"] = df["Close"].rolling(window=30).mean()
    df["Volatility"] = df["Return"].rolling(window=10).std()

    # 3. Variable objetivo: 1 si el precio sube al día siguiente, 0 si baja
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

    # Eliminamos la columna auxiliar "Next_Close" para no tener data leakage (fuga de información)
    df.drop(columns=["Next_Close"], inplace=True)

    # Limpieza de valores nulos (NaN) generados por los desplazamientos y ventanas moviles
    df = df.dropna().reset_index(drop=True)

    # Asegurar que el directorio de salida exista
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Guardar caracteristicas tabulares procesadas
    output_path = "data/processed/sp500_caracteristicas.parquet"
    df.to_parquet(output_path)
    print(f"Caracteristicas procesadas guardadas exitosamente en '{output_path}'.")

    return df

if __name__ == "__main__":
    RAW_DATA = "data/raw/sp500_diario.parquet"

    crear_caracteristicas_financieras(input_path=RAW_DATA)
