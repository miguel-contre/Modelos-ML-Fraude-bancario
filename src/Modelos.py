import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# ===========================
# 1. División de datos
# ===========================
def preparar_datos(df, target_col="Class", test_size = 0.2, random_state = 42, balance = False):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# ======================
# 1.1 Opcional para etiquetas desbalanciadas
# ======================

def smote(X_train , y_train , random_state = 42):
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res


# ===========================
# 2. Modelos
# ===========================
def entrenar_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def entrenar_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model

def entrenar_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), # balance interno
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def evaluar_modelo(model, X_test, y_test):
    # 1. Predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # 2. Reporte de métricas
    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # 3. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicho 0", "Predicho 1"],
                yticklabels=["Real 0", "Real 1"])
    plt.title("Matriz de confusión")
    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.show()
