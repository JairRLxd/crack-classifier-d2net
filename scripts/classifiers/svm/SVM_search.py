import csv
import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Edita aqui las rutas y opciones
TRAIN_BOW_DIR = "outputs/KmeansResults/K800T2"
OUT_DIR = os.path.join(TRAIN_BOW_DIR, "svm_search")

# Preprocesamiento
USE_SCALER = True
SCALER_WITH_MEAN = False

# Grid de hiperparametros
PARAM_GRID = [
    {
        "scaler": ["passthrough", StandardScaler(with_mean=False), StandardScaler(with_mean=True)],
        "svc__kernel": ["linear"],
        "svc__C": [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3],
        "svc__class_weight": [None, "balanced"],
        "svc__tol": [1e-2, 1e-3, 1e-4],
        "svc__shrinking": [True, False],
        "svc__decision_function_shape": ["ovr"],
        "svc__break_ties": [False],
        "svc__max_iter": [-1],
    },
    {
        "scaler": ["passthrough", StandardScaler(with_mean=False), StandardScaler(with_mean=True)],
        "svc__kernel": ["rbf"],
        "svc__C": [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3],
        "svc__gamma": ["scale", "auto", 1e-4, 1e-3, 1e-2, 1e-1],
        "svc__class_weight": [None, "balanced"],
        "svc__tol": [1e-2, 1e-3, 1e-4],
        "svc__shrinking": [True, False],
        "svc__decision_function_shape": ["ovr"],
        "svc__break_ties": [False],
        "svc__max_iter": [-1],
    },
    {
        "scaler": ["passthrough", StandardScaler(with_mean=False), StandardScaler(with_mean=True)],
        "svc__kernel": ["poly"],
        "svc__C": [1e-1, 1.0, 10.0, 100.0],
        "svc__degree": [2, 3, 4, 5],
        "svc__gamma": ["scale", "auto", 1e-3, 1e-2],
        "svc__coef0": [0.0, 0.5, 1.0, 2.0],
        "svc__class_weight": [None, "balanced"],
        "svc__tol": [1e-3, 1e-4],
        "svc__shrinking": [True, False],
        "svc__decision_function_shape": ["ovr"],
        "svc__break_ties": [False],
        "svc__max_iter": [-1],
    },
    {
        "scaler": ["passthrough", StandardScaler(with_mean=False), StandardScaler(with_mean=True)],
        "svc__kernel": ["sigmoid"],
        "svc__C": [1e-2, 1e-1, 1.0, 10.0, 100.0],
        "svc__gamma": ["scale", "auto", 1e-4, 1e-3, 1e-2],
        "svc__coef0": [-1.0, 0.0, 0.5, 1.0],
        "svc__class_weight": [None, "balanced"],
        "svc__tol": [1e-2, 1e-3, 1e-4],
        "svc__shrinking": [True, False],
        "svc__decision_function_shape": ["ovr"],
        "svc__break_ties": [False],
        "svc__max_iter": [-1],
    },
]

# Configuracion de validacion
CV_SPLITS = 5
SHUFFLE = True
RANDOM_STATE = 42
N_JOBS = -1
VERBOSE = 2
REPORT_ZERO_DIVISION = 0
RETURN_TRAIN_SCORE = True
REFIT = True
SCORING = "f1_macro"
CACHE_SIZE = 512
PROBABILITY = False
VERBOSE_SVC = False

RESULT_FIELDNAMES = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "std_train_score",
    "mean_fit_time",
    "mean_score_time",
    "param_scaler",
    "param_svc__kernel",
    "param_svc__C",
    "param_svc__gamma",
    "param_svc__degree",
    "param_svc__coef0",
    "param_svc__class_weight",
    "param_svc__tol",
    "param_svc__shrinking",
    "param_svc__decision_function_shape",
    "param_svc__break_ties",
    "param_svc__max_iter",
]


def build_estimator():
    svm = SVC(
        probability=PROBABILITY,
        cache_size=CACHE_SIZE,
        verbose=VERBOSE_SVC,
        random_state=RANDOM_STATE,
    )

    scaler_step = StandardScaler(with_mean=SCALER_WITH_MEAN) if USE_SCALER else "passthrough"
    return Pipeline(
        steps=[
            ("scaler", scaler_step),
            ("svc", svm),
        ]
    )


def get_total_combinations():
    return sum(len(ParameterGrid(grid)) for grid in PARAM_GRID)


def to_python_value(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if value is np.ma.masked:
        return ""
    return value


def simplify_best_params(best_params):
    cleaned = {}
    for key, value in best_params.items():
        if key == "scaler":
            if value == "passthrough":
                cleaned["scaler"] = "passthrough"
            else:
                cleaned["scaler"] = value.__class__.__name__
                cleaned["scaler_with_mean"] = bool(getattr(value, "with_mean", False))
            continue
        if key.startswith("svc__"):
            cleaned[key.replace("svc__", "", 1)] = to_python_value(value)
        else:
            cleaned[key] = to_python_value(value)
    return cleaned


def build_cv_rows(search):
    rows = []
    cv_results = search.cv_results_

    for i in range(len(cv_results["params"])):
        row = {}
        for field in RESULT_FIELDNAMES:
            value = cv_results.get(field, "")
            if isinstance(value, (list, np.ndarray, np.ma.MaskedArray)):
                value = value[i]
            row[field] = to_python_value(value)
        rows.append(row)

    rows.sort(key=lambda row: (row["rank_test_score"], -row["mean_test_score"]))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    x_train = np.load(os.path.join(TRAIN_BOW_DIR, "histograms.npy"))
    y_train = np.load(os.path.join(TRAIN_BOW_DIR, "image_labels.npy"), allow_pickle=True)

    print(f"x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"Evaluando {get_total_combinations()} combinaciones con {CV_SPLITS}-fold CV...")

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=SHUFFLE,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=build_estimator(),
        param_grid=PARAM_GRID,
        scoring=make_scorer(f1_score, average="macro"),
        n_jobs=N_JOBS,
        cv=cv,
        refit=REFIT,
        verbose=VERBOSE,
        return_train_score=RETURN_TRAIN_SCORE,
    )
    search.fit(x_train, y_train)

    rows = build_cv_rows(search)
    with open(os.path.join(OUT_DIR, "grid_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    best_params = simplify_best_params(search.best_params_)
    best_config = {
        **best_params,
        "best_cv_macro_f1": float(search.best_score_),
        "cv_splits": CV_SPLITS,
        "shuffle": SHUFFLE,
        "random_state": RANDOM_STATE,
        "use_scaler": USE_SCALER,
        "scaler_with_mean": SCALER_WITH_MEAN,
        "cache_size": CACHE_SIZE,
        "probability": PROBABILITY,
        "scoring": SCORING,
        "total_combinations": get_total_combinations(),
    }

    with open(os.path.join(OUT_DIR, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    y_cv_pred = cross_val_predict(
        search.best_estimator_,
        x_train,
        y_train,
        cv=cv,
        n_jobs=N_JOBS,
    )

    labels = np.unique(y_train)
    cm = confusion_matrix(y_train, y_cv_pred, labels=labels)
    report = classification_report(
        y_train,
        y_cv_pred,
        zero_division=REPORT_ZERO_DIVISION,
    )

    with open(os.path.join(OUT_DIR, "best_validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    np.save(os.path.join(OUT_DIR, "best_cv_confusion_matrix.npy"), cm)
    np.save(os.path.join(OUT_DIR, "best_cv_confusion_matrix_labels.npy"), labels)
    np.save(os.path.join(OUT_DIR, "best_cv_y_true.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "best_cv_y_pred.npy"), y_cv_pred)

    print("\nMejor configuracion encontrada:")
    print(best_config)
    print("\nReporte CV de la mejor configuracion:\n")
    print(report)
    print("labels:", labels.tolist())
    print("confusion_matrix:\n", cm)
    print(f"Resultados guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
