import kagglehub
import shutil
from pathlib import Path

# Descarga el dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path de descarga por defecto:", path)

# Definir destino
destino = Path(r"C:\Users\migue\OneDrive\Escritorio\Fraude_bancario\data") 

# Mover todos los archivos descargados
for archivo in Path(path).glob("*"):
    shutil.copy(archivo, destino)  # usa .move si quieres mover en vez de copiar

print(f"Archivos copiados a: {destino.resolve()}")
