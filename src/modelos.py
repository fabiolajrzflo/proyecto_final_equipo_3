"""Modelos clásicos (Módulo 3): pipelines, espacios de búsqueda y los dos pasos por modelo
del diagrama: selección de variables (RFECV) -> búsqueda de hiperparámetros.

Imputación y escalado viven DENTRO del Pipeline: en cada fold se ajustan sólo con su train.
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config

RS = config.RANDOM_STATE


def _imputador():
    # keep_empty_features: no borra columnas vacías en un fold (RFECV necesita columnas fijas)
    return SimpleImputer(strategy="median", keep_empty_features=True)


def crear_pipeline(nombre, rapido=False):
    """Pipeline con hiperparámetros base. rapido=True: menos árboles (sólo para el RFECV)."""
    if nombre == "Regresión Logística":
        modelo = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=RS)
        return Pipeline([("imputer", _imputador()), ("scaler", StandardScaler()), ("modelo", modelo)])
    if nombre == "Random Forest":
        modelo = RandomForestClassifier(
            n_estimators=100 if rapido else 300, min_samples_leaf=20, max_features="sqrt",
            max_samples=0.5 if rapido else None, class_weight="balanced_subsample",
            random_state=RS, n_jobs=-1)
        return Pipeline([("imputer", _imputador()), ("modelo", modelo)])
    if nombre == "XGBoost":
        # XGBoost maneja los faltantes de forma nativa y no necesita escalado
        modelo = XGBClassifier(
            n_estimators=150 if rapido else 300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=5, tree_method="hist",
            eval_metric="aucpr", random_state=RS, n_jobs=-1)
        return Pipeline([("modelo", modelo)])
    raise ValueError(nombre)


IMPORTANCIA = {
    "Regresión Logística": "named_steps.modelo.coef_",
    "Random Forest": "named_steps.modelo.feature_importances_",
    "XGBoost": "named_steps.modelo.feature_importances_",
}

ESPACIOS = {
    "Regresión Logística": {
        "modelo__C": loguniform(1e-3, 1e1),
        "modelo__penalty": ["l1", "l2"],
        "modelo__solver": ["liblinear"],
    },
    "Random Forest": {
        "modelo__n_estimators": [200, 300],
        "modelo__max_depth": [None, 8, 12, 16],
        "modelo__min_samples_leaf": [5, 10, 20, 50, 100],
        "modelo__max_features": ["sqrt", 0.3, 0.5],
        "modelo__max_samples": [0.5, 0.8, None],
        "modelo__class_weight": ["balanced_subsample", None],
    },
    "XGBoost": {
        "modelo__n_estimators": [200, 300, 500],
        "modelo__learning_rate": loguniform(0.02, 0.2),
        "modelo__max_depth": [3, 4, 5, 6],
        "modelo__min_child_weight": [1, 5, 10, 20],
        "modelo__subsample": [0.6, 0.8, 1.0],
        "modelo__colsample_bytree": [0.5, 0.7, 0.9],
        "modelo__reg_lambda": loguniform(0.1, 10),
        "modelo__scale_pos_weight": [1, 3, 6, 12],
    },
}

# Paralelismo: la logística es de un solo hilo (paralelizamos folds); los árboles ya usan todos los núcleos
N_JOBS_CV = {"Regresión Logística": 5, "Random Forest": 1, "XGBoost": 1}


# --------------------------------------------------------------- Paso 1: selección de variables
def seleccion_rfecv(nombre, X, y, folds, step=0.1, min_features=5):
    selector = RFECV(
        estimator=crear_pipeline(nombre, rapido=True), importance_getter=IMPORTANCIA[nombre],
        step=step, cv=folds, scoring="average_precision", min_features_to_select=min_features,
        n_jobs=N_JOBS_CV[nombre],
    )
    selector.fit(X, y)
    variables = list(X.columns[selector.support_])
    curva = pd.DataFrame({
        "n_variables": selector.cv_results_["n_features"],
        "pr_auc_cv": selector.cv_results_["mean_test_score"],
        "pr_auc_cv_desv": selector.cv_results_["std_test_score"],
    })
    return variables, curva


# --------------------------------------------------------------- Paso 2: hiperparámetros
def busqueda_hiperparametros(nombre, X, y, folds, n_iter):
    busqueda = RandomizedSearchCV(
        crear_pipeline(nombre), ESPACIOS[nombre], n_iter=n_iter, scoring="average_precision",
        cv=folds, refit=False, return_train_score=True, random_state=RS, n_jobs=N_JOBS_CV[nombre],
    )
    busqueda.fit(X, y)
    res = pd.DataFrame(busqueda.cv_results_)
    cols = [c for c in res.columns if c.startswith("param_")] + [
        "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    res = res[cols].sort_values("rank_test_score").reset_index(drop=True)
    return busqueda.best_params_, res


def pipeline_con_parametros(nombre, params):
    p = crear_pipeline(nombre)
    p.set_params(**params)
    return p


# --------------------------------------------------------------- Guardado de artefactos
def _a_json(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def guardar_artefactos(nombre, variables, params, cv_folds, oof, umbral, modelo_final, extra=None):
    carpeta = config.MODELOS / config.CLAVES_MODELO[nombre]
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "best_features.json").write_text(json.dumps(variables, indent=2, ensure_ascii=False))
    (carpeta / "best_params.json").write_text(
        json.dumps({k: _a_json(v) for k, v in params.items()}, indent=2, ensure_ascii=False))
    cv_folds.to_csv(carpeta / "cv_folds.csv", index=False)
    oof.to_parquet(carpeta / "oof.parquet", index=False)
    resumen = {
        "modelo": nombre,
        "n_variables": len(variables),
        "umbral": float(umbral),
        "pr_auc_val_prom": float(cv_folds["PR-AUC val"].mean()),
        "pr_auc_val_desv": float(cv_folds["PR-AUC val"].std()),
        "pr_auc_train_prom": float(cv_folds["PR-AUC train"].mean()),
        "roc_auc_val_prom": float(cv_folds["ROC-AUC val"].mean()),
        "roc_auc_train_prom": float(cv_folds["ROC-AUC train"].mean()),
    }
    resumen.update(extra or {})
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    if modelo_final is not None:
        joblib.dump(modelo_final, carpeta / "modelo.joblib")
    return carpeta
