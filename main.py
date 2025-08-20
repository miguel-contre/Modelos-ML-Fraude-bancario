from src.analysis import cargar_datos , resumen
from src.Modelos import preparar_datos , smote , entrenar_logistic , entrenar_random_forest ,  entrenar_xgboost , evaluar_modelo
def main():
    # Ruta a tus datos (ajústala según tu carpeta destino)
    DATA_PATH = r"C:\Users\migue\OneDrive\Escritorio\Fraude_bancario\data\creditcard.csv"

    # 1. Cargar
    df = cargar_datos(DATA_PATH)

    # 2. Resumen
    resumen(df, "Class")

    #Separa datos de entramiento y test 
    X_train , X_test , y_train , y_test = preparar_datos(df, target_col="Class", test_size = 0.2, random_state = 42)

    model_logis = entrenar_logistic(X_train , y_train)
    model_forest = entrenar_random_forest(X_train , y_train)
    model_xgb = entrenar_xgboost(X_train , y_train)

    evaluar_modelo(model = model_logis , X_test = X_test , y_test = y_test)
    evaluar_modelo(model = model_forest , X_test = X_test , y_test = y_test) 
    evaluar_modelo(model = model_xgb , X_test = X_test , y_test = y_test)

if __name__ == "__main__":
    main()
