import csv
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict


# Edita aqui las rutas y opciones
TRAIN_BOW_DIR = "outputs/KmeansResults/K200T2"
OUT_DIR = os.path.join(TRAIN_BOW_DIR, "rf_search")

# Grid de hiperparametros
PARAM_GRID = {
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [20, 30],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
    "criterion": ["gini", "entropy", "log_loss"],
    "bootstrap": [True, False],
}

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

RESULT_FIELDNAMES = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "std_train_score",
    "mean_fit_time",
    "mean_score_time",
    "param_n_estimators",
    "param_max_depth",
    "param_min_samples_leaf",
    "param_max_features",
    "param_min_samples_split",
    "param_criterion",
    "param_bootstrap",
]


def build_estimator():
    return RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def get_total_combinations():
    return len(ParameterGrid(PARAM_GRID))


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
    return {key: to_python_value(value) for key, value in best_params.items()}


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
