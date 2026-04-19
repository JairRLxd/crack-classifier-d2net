import json
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# Edita aqui las rutas y opciones
MODEL_DIR = "outputs/KmeansResults/K500T2/rf_500T2"
TEST_BOW_DIR = "outputs/KMEANSTest/K500T2"
OUT_DIR = "outputs/RF_eval/bow_test/rf_eval"
REPORT_ZERO_DIVISION = 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Cargar modelo entrenado
    model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    model_config_path = os.path.join(MODEL_DIR, "model_config.json")
    clf = joblib.load(model_path)
    model_config = None

    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)

    # 2) Cargar test
    x_test = np.load(os.path.join(TEST_BOW_DIR, "histograms_test.npy"))
    y_test = np.load(os.path.join(TEST_BOW_DIR, "image_labels_test.npy"), allow_pickle=True)

    print(f"x_test={x_test.shape}, y_test={y_test.shape}")

    # 3) Predecir y evaluar
    y_pred = clf.predict(x_test)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test,
        y_pred,
        zero_division=REPORT_ZERO_DIVISION,
    )

    print(report)
    print("labels:", labels.tolist())
    print("confusion_matrix:\n", cm)
    if model_config is not None:
        print("Configuracion del modelo evaluado:")
        print(model_config)

    # 4) Guardar resultados
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    if model_config is not None:
        with open(os.path.join(OUT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

    np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), cm)
    np.save(os.path.join(OUT_DIR, "confusion_matrix_labels.npy"), labels)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred)
    np.save(os.path.join(OUT_DIR, "feature_importances.npy"), clf.feature_importances_)

    print(f"Modelo cargado desde: {model_path}")
    print(f"Evaluacion guardada en: {OUT_DIR}")


if __name__ == "__main__":
    main()
