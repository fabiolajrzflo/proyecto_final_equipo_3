"""Métricas, umbral de decisión y evaluación con el K-Fold temporal."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from . import config


def umbral_optimo(y_true, prob):
    """Umbral sobre la curva Precision-Recall que maximiza el F1 de la clase 1."""
    p, r, t = precision_recall_curve(y_true, prob)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.nanargmax(f1[:-1])])   # el último punto de la curva no tiene umbral


def metricas(y_true, prob, umbral):
    pred = (np.asarray(prob) >= umbral).astype(int)
    return {
        "PR-AUC": average_precision_score(y_true, prob),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, pred),
    }


def evaluar_cv(ajustar_predecir, X, y, folds):
    """Corre el K-Fold temporal y devuelve (tabla por fold, predicciones fuera de fold, umbral).

    ajustar_predecir(X_tr, y_tr, X_va) -> (prob_tr, prob_va). Se mide en train Y en validación
    para diagnosticar sobreajuste (train_avg vs val_avg del diagrama). El sobreajuste se lee con
    ROC-AUC: el PR-AUC depende de la tasa de tardíos, que cambia entre train y validación.
    """
    y = pd.Series(y).reset_index(drop=True)
    filas, oof = [], []
    for k, (tr, va) in enumerate(folds, 1):
        prob_tr, prob_va = ajustar_predecir(X.iloc[tr], y.iloc[tr], X.iloc[va])
        filas.append({"Fold": k, "PR-AUC train": average_precision_score(y.iloc[tr], prob_tr),
                      "PR-AUC val": average_precision_score(y.iloc[va], prob_va),
                      "ROC-AUC train": roc_auc_score(y.iloc[tr], prob_tr),
                      "ROC-AUC val": roc_auc_score(y.iloc[va], prob_va),
                      "% tardíos train": y.iloc[tr].mean() * 100,
                      "% tardíos val": y.iloc[va].mean() * 100})
        oof.append(pd.DataFrame({"idx": va, "fold": k, "y": y.iloc[va].to_numpy(), "prob": prob_va}))
    oof = pd.concat(oof, ignore_index=True)
    umbral = umbral_optimo(oof["y"], oof["prob"])
    tabla = pd.DataFrame(filas)
    # Precision / Recall / F1 por fold con el umbral elegido en TODAS las predicciones fuera de fold
    for k, g in oof.groupby("fold"):
        m = metricas(g["y"], g["prob"], umbral)
        for met in ("Precision", "Recall", "F1"):
            tabla.loc[tabla["Fold"] == k, f"{met} val"] = m[met]
    return tabla, oof, umbral


def ajustar_predecir_sklearn(pipeline):
    def _f(X_tr, y_tr, X_va):
        m = clone(pipeline).fit(X_tr, y_tr)
        return m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1]
    return _f


def resumen_cv(tabla):
    """Promedio ± desviación de cada métrica entre folds."""
    cols = [c for c in tabla.columns if c != "Fold"]
    return pd.DataFrame({"promedio": tabla[cols].mean(), "desv": tabla[cols].std()}).T


def captura_top_k(y_true, prob, fracciones=(0.05, 0.10, 0.20, 0.30)):
    """Si revisamos el k% de pedidos con mayor riesgo, ¿qué % de los tardíos capturamos?"""
    orden = np.argsort(-np.asarray(prob))
    y = np.asarray(y_true)[orden]
    total = y.sum()
    filas = []
    for f in fracciones:
        n = int(round(len(y) * f))
        capt = y[:n].sum()
        filas.append({"% pedidos revisados": f * 100, "% tardíos capturados": capt / total * 100,
                      "Precision": capt / n, "Lift": (capt / n) / y.mean()})
    return pd.DataFrame(filas)


def mejor_modelo(pr_auc_cv, orden_simple=None, tol=config.TOLERANCIA_EMPATE):
    """Mayor PR-AUC del CV; si dos quedan a menos de `tol`, gana el más simple."""
    orden_simple = orden_simple or list(pr_auc_cv)
    mejor = max(pr_auc_cv.values())
    return next(n for n in orden_simple if pr_auc_cv[n] >= mejor - tol)
