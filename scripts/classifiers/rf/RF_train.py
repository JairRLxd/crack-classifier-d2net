import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Edita aqui las rutas y opciones
TRAIN_BOW_DIR = "outputs/KmeansResults/K500T2"
MODEL_DIR = os.path.join(TRAIN_BOW_DIR, "rf_500T2")

# Hiperparametros
N_ESTIMATORS = 300      
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"
MIN_SAMPLES_SPLIT = 2
CRITERION = "entropy"
BOOTSTRAP = False
RANDOM_STATE = 42
N_JOBS = -1


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Cargar train
    x_train = np.load(os.path.join(TRAIN_BOW_DIR, "histograms.npy"))
    y_train = np.load(os.path.join(TRAIN_BOW_DIR, "image_labels.npy"), allow_pickle=True)

    print(f"x_train={x_train.shape}, y_train={y_train.shape}")

    # 2) Entrenar RF
    model_config = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": MAX_FEATURES,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "criterion": CRITERION,
        "bootstrap": BOOTSTRAP,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    }

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        min_samples_split=MIN_SAMPLES_SPLIT,
        criterion=CRITERION,
        bootstrap=BOOTSTRAP,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    clf.fit(x_train, y_train)

    # 3) Guardar modelo entrenado
    joblib.dump(clf, os.path.join(MODEL_DIR, "random_forest.joblib"))
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    np.save(os.path.join(MODEL_DIR, "classes.npy"), clf.classes_)
    np.save(os.path.join(MODEL_DIR, "feature_importances.npy"), clf.feature_importances_)

    print("Configuracion del modelo:")
    print(model_config)
    print(f"Modelo guardado en: {MODEL_DIR}")


if __name__ == "__main__":
    main()
