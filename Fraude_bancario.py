import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

#Configuración del usuario
DATA_PATH = r"C:\Users\migue\.cache\kagglehub\datasets\mlg-ulb\creditcardfraud\versions\3\creditcard.csv" 
TARGET_OVERRIDE = None

#Carga robusta
path = Path(DATA_PATH)
assert path.exists(), f"No se encontró el archivo en: {path}" #saber si existe

#Detectar separador
with open(path, "rb") as f:
    head = f.read(2048)
text_sample = head.decode(errors="ignore")
sep = "," if text_sample.count(",") >= text_sample.count(";") else ";"

df = pd.read_csv(path, sep=sep)
print(f"Archivo cargado: {path.name} | filas={len(df):,} | columnas={df.shape[1]} | sep='{sep}'")
print(df.head(3))

candidates = ["fraud", "is_fraud", "es_fraude", "fraude", "target", "label", "y"]
if TARGET_OVERRIDE is not None:
    target_col = TARGET_OVERRIDE
else:
    lower_map = {c.lower(): c for c in df.columns}
    found = [lower_map[c] for c in lower_map.keys() if c in candidates]
    target_col = found[0] if found else None

#Checks 
#Dimensiones
print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")

#Tipos
dtype_summary = df.dtypes.to_frame("dtype")
print(dtype_summary.T)

#Cardinalidad (número de valores únicos) por columna
card = df.nunique(dropna=True).sort_values(ascending=False).to_frame("n_unique")
print(card.head())

#Columnas no numericas y numericas
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
cat_cols = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c]))]
print(f"Cantidad de variables numericas{num_cols}")
print(f"Cantidad de variables no numericas{cat_cols}")

#Estadisticos
if num_cols:
    desc = (df[num_cols]
            .describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
            .T.sort_values("std", ascending=False))
    print(desc.head())


#Nulos
nulls = df.isna().sum().sort_values(ascending=False)
nulls_pct = (nulls / len(df)).round(4)
missing_table = pd.DataFrame({"n_nulls": nulls, "pct_nulls": nulls_pct})
print(missing_table[missing_table.n_nulls > 0].head())

# Balance de clases
vc = df["Class"].value_counts(dropna=False).rename({0:"No Fraude (0)", 1:"Fraude (1)"})
vc_pct = (vc / vc.sum() * 100).round(2)
balance = pd.DataFrame({"conteo": vc, "porcentaje": vc_pct})
print(balance)

# Gráfico simple de balance
ax = balance["conteo"].plot(kind="bar", rot=0, title="Balance de clases (conteo)")
ax.bar_label(ax.containers[0])
plt.show()

# Tasa de fraude
if 1 in df["Class"].unique():
    fraud_rate = (df["Class"] == 1).mean()
    print(f"Tasa de fraude: {fraud_rate:.4%}")



# Histograma de todas las variables numéricas (subplots)
df[num_cols].hist(bins=10, figsize=(30, 30))
plt.suptitle("Distribuciones de variables numéricas", fontsize=20)
plt.show()

# Histograma de la variable Amount diferenciando fraude vs no fraude
plt.figure(figsize=(8,5))
sns.histplot(data=df, x="Amount", hue="Class", bins=50, log_scale=(True, False))
plt.title("Distribución de montos según clase")
plt.show()

print(df)
