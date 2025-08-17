from src.analysis import cargar_datos, resumen

def main():
    # Ruta a tus datos (ajústala según tu carpeta destino)
    DATA_PATH = r"C:\Users\migue\OneDrive\Escritorio\Fraude_bancario\data\creditcard.csv"

    # 1. Cargar
    df = cargar_datos(DATA_PATH)

    # 2. Resumen
    resumen(df, "Class")


if __name__ == "__main__":
    main()
