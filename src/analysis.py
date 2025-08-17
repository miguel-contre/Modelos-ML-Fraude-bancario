import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def cargar_datos(path: str):
    #Configuración del usuario
    """Carga un CSV detectando el separador automáticamente"""
    path = Path(path)
    assert path.exists(), f"No se encontró el archivo en: {path}"

    #Arbre el archivo y ve su separador
    with open(path, "rb") as f:
        head = f.read(2048)
    text_sample = head.decode(errors="ignore")
    sep = "," if text_sample.count(",") >= text_sample.count(";") else ";"
    
    #Carga los datos
    df = pd.read_csv(path, sep=sep)
    print(f"Archivo cargado: {path.name} | filas={len(df):,} | columnas={df.shape[1]} | sep='{sep}'")

    return df

def resumen(df: pd.DataFrame, target_col: str = None):
    """Muestra estadísticas y checks básicos"""
    #Dimensiinalidad 
    print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
    print(df.dtypes.to_frame("dtype").T)

    #Cardianalidad de datos
    card = df.nunique(dropna=True).sort_values(ascending=False).to_frame("n_unique")
    print(card.head())

    #Columnas no numericas y numericas
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    print(f"Cantidad de variables numéricas: {len(num_cols)}")
    print(f"Cantidad de variables no numéricas: {len(cat_cols)}")

    #Estadisticos 
    if num_cols:
        desc = (df[num_cols]
                .describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
                .T.sort_values("std", ascending=False))
        print(desc.head())

    #Nulos
    nulls = df.isna().sum().sort_values(ascending=False)
    missing_table = pd.DataFrame({"n_nulls": nulls, "pct_nulls": (nulls / len(df)).round(4)})
    print(missing_table[missing_table.n_nulls > 0].head())

    # Balance de clases
    vc = df[target_col].value_counts(dropna=False).rename({0:"No Fraude (0)", 1:"Fraude (1)"})
    balance = pd.DataFrame({"conteo": vc, "porcentaje": (vc / vc.sum() * 100).round(2)})
    print(balance)

    # Tasa de fraude
    if 1 in df[target_col].unique():
        fraud_rate = (df[target_col] == 1).mean()
        print(f"Tasa de fraude: {fraud_rate:.4%}")


