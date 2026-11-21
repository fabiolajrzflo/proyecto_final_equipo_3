"""Etapa 4 del flujo: corte de test y K-Fold temporal con PURGA.

Purga: un pedido sólo puede entrenar un modelo que se usará a partir del instante T si su
label ya se conocía antes de T (`label_conocida_en < T`). Sin esto, los últimos pedidos del
entrenamiento tendrían labels que en la vida real todavía no se sabrían.

    |------ train (purgado) ------|~~ purga ~~|------ validación / test ------|
"""
import numpy as np
import pandas as pd


def corte_test(df, fecha_corte):
    """Índices (posicionales) de train y test. df debe estar en orden cronológico."""
    T = pd.Timestamp(fecha_corte)
    compra, conocida = df["order_purchase_timestamp"], df["label_conocida_en"]
    es_test = (compra >= T).to_numpy()
    es_train = ((compra < T) & (conocida < T)).to_numpy()
    purgados = int(((compra < T) & (conocida >= T)).sum())
    return np.flatnonzero(es_train), np.flatnonzero(es_test), purgados


def folds_temporales(compra, conocida, n_folds):
    """K-Fold temporal de ventana creciente (como TimeSeriesSplit) con purga.

    El train se divide en n_folds + 1 bloques consecutivos del mismo tamaño.
    Fold k: valida con el bloque k+1 y entrena con los bloques 1..k cuyos labels ya se
    conocían al inicio de la validación. Devuelve una lista [(idx_train, idx_val), ...]
    con posiciones relativas al train, lista para pasar como `cv=` a scikit-learn.
    """
    compra = pd.Series(compra).reset_index(drop=True)
    conocida = pd.Series(conocida).reset_index(drop=True)
    assert compra.is_monotonic_increasing, "el train debe estar en orden cronológico"
    bloques = np.array_split(np.arange(len(compra)), n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        val = bloques[k]
        inicio_val = compra.iloc[val[0]]
        candidatos = np.concatenate(bloques[:k])
        tr = candidatos[(conocida.iloc[candidatos] < inicio_val).to_numpy()]
        folds.append((tr, val))
    return folds


def resumen_folds(folds, compra, y):
    compra = pd.Series(compra).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    filas = []
    for k, (tr, va) in enumerate(folds, 1):
        filas.append({
            "Fold": k,
            "Train desde": compra.iloc[tr[0]].date(), "Train hasta": compra.iloc[tr[-1]].date(),
            "Filas train": len(tr),
            "Val desde": compra.iloc[va[0]].date(), "Val hasta": compra.iloc[va[-1]].date(),
            "Filas val": len(va),
            "% tardíos train": round(y.iloc[tr].mean() * 100, 2),
            "% tardíos val": round(y.iloc[va].mean() * 100, 2),
        })
    return pd.DataFrame(filas).set_index("Fold")


def verificar_sin_fuga(folds, compra, conocida):
    """Comprueba que ningún label de entrenamiento se conocía después del inicio de su validación."""
    compra = pd.Series(compra).reset_index(drop=True)
    conocida = pd.Series(conocida).reset_index(drop=True)
    for tr, va in folds:
        assert conocida.iloc[tr].max() < compra.iloc[va].min()
        assert compra.iloc[tr].max() < compra.iloc[va].min()
    return True
