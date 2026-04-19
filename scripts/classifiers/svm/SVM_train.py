import json
import os

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Edita aqui las rutas y opciones
TRAIN_BOW_DIR = "outputs/KmeansResults/K800T2"
MODEL_DIR = os.path.join(TRAIN_BOW_DIR, "svm_800T2")

# Preprocesamiento
USE_SCALER = False
SCALER_WITH_MEAN = False

# Hiperparametros SVM
C = 10.0
KERNEL = "poly"
DEGREE = 4
GAMMA = "scale"
COEF0 = 0.0
SHRINKING = True
PROBABILITY = False
TOL = 1e-3
CACHE_SIZE = 512
CLASS_WEIGHT = None
VERBOSE = False
MAX_ITER = -1
DECISION_FUNCTION_SHAPE = "ovr"
BREAK_TIES = False
RANDOM_STATE = 42



def build_estimator():
    svm = SVC(
        C=C,
        kernel=KERNEL,
        degree=DEGREE,
        gamma=GAMMA,
        coef0=COEF0,
        shrinking=SHRINKING,
        probability=PROBABILITY,
        tol=TOL,
        cache_size=CACHE_SIZE,
        class_weight=CLASS_WEIGHT,
        verbose=VERBOSE,
        max_iter=MAX_ITER,
        decision_function_shape=DECISION_FUNCTION_SHAPE,
        break_ties=BREAK_TIES,
        random_state=RANDOM_STATE,
    )

    scaler_step = StandardScaler(with_mean=SCALER_WITH_MEAN) if USE_SCALER else "passthrough"
    return Pipeline(
        steps=[
            ("scaler", scaler_step),
            ("svc", svm),
        ]
    )


def get_model_config():
    return {
        "train_bow_dir": TRAIN_BOW_DIR,
        "model_dir": MODEL_DIR,
        "scaler": "StandardScaler" if USE_SCALER else "passthrough",
        "use_scaler": USE_SCALER,
        "scaler_with_mean": SCALER_WITH_MEAN,
        "C": C,
        "kernel": KERNEL,
        "degree": DEGREE,
        "gamma": GAMMA,
        "coef0": COEF0,
        "shrinking": SHRINKING,
        "probability": PROBABILITY,
        "tol": TOL,
        "cache_size": CACHE_SIZE,
        "class_weight": CLASS_WEIGHT,
        "verbose": VERBOSE,
        "max_iter": MAX_ITER,
        "decision_function_shape": DECISION_FUNCTION_SHAPE,
        "break_ties": BREAK_TIES,
        "random_state": RANDOM_STATE,
    }


def get_svc_from_estimator(estimator):
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["svc"]
    return estimator


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    x_train = np.load(os.path.join(TRAIN_BOW_DIR, "histograms.npy"))
    y_train = np.load(os.path.join(TRAIN_BOW_DIR, "image_labels.npy"), allow_pickle=True)

    print(f"x_train={x_train.shape}, y_train={y_train.shape}")

    model_config = get_model_config()
    clf = build_estimator()
    clf.fit(x_train, y_train)

    svm = get_svc_from_estimator(clf)

    joblib.dump(clf, os.path.join(MODEL_DIR, "svm.joblib"))
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    np.save(os.path.join(MODEL_DIR, "classes.npy"), svm.classes_)
    np.save(os.path.join(MODEL_DIR, "support_per_class.npy"), svm.n_support_)

    print("Configuracion del modelo:")
    print(model_config)
    print("Soportes por clase:", svm.n_support_.tolist())
    print(f"Modelo guardado en: {MODEL_DIR}")


if __name__ == "__main__":
    main()
