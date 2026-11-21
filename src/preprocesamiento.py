"""Codificación de categóricas (ajustada sólo con train) y pipelines de imputación/escalado.

- One-hot: el codificador aprende la lista de categorías (las MAX_CATEGORIAS más frecuentes)
  SÓLO con el train; las demás caen en "infrequent". No usa el label.
- Imputación y escalado: van DENTRO del Pipeline de cada modelo, así que se ajustan en cada
  fold sólo con los datos de entrenamiento de ese fold.
"""
import re

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from . import config

COLUMNAS_CATEGORICAS = [
    "customer_state", "seller_state", "main_payment_type", "main_category",
    "purchase_weekday", "franja_horaria",
]

# Identificadores, fechas y el label: nunca entran al modelo
COLUMNAS_EXCLUIR = [
    "order_id", "customer_id", "customer_zip_code_prefix", "main_seller_id", "ruta",
    "order_purchase_timestamp", "label_conocida_en", "delivery_late",
]


def ajustar_codificador(df_train):
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=config.MAX_CATEGORIAS,
                        sparse_output=False, dtype=int)
    enc.fit(df_train[COLUMNAS_CATEGORICAS].astype(str))
    return enc


def _limpiar_nombre(c):
    return re.sub(r"[\[\]<>,\s]", "_", c)


def construir_X(df, enc):
    numericas = [c for c in df.columns if c not in COLUMNAS_CATEGORICAS + COLUMNAS_EXCLUIR]
    dummies = pd.DataFrame(
        enc.transform(df[COLUMNAS_CATEGORICAS].astype(str)),
        columns=[_limpiar_nombre(c) for c in enc.get_feature_names_out(COLUMNAS_CATEGORICAS)],
        index=df.index,
    )
    X = pd.concat([df[numericas].astype(float), dummies], axis=1)
    return X.reset_index(drop=True)
