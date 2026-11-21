"""Etapa 1 del flujo: carga y limpieza de las tablas de Olist + definición del label."""
import numpy as np
import pandas as pd

from . import config

ESTADOS_BRASIL = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA",
    "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
]

ARCHIVOS = {
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
}

COLS_DIM = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]


def cargar_tablas():
    """Lee los CSV originales desde datos/ (sin descargar nada)."""
    return {k: pd.read_csv(config.DATOS_RAW / v) for k, v in ARCHIVOS.items()}


def _zip5(s):
    return s.astype(str).str.zfill(5)


def limpiar_customers(df):
    df = df.copy()
    df["customer_zip_code_prefix"] = _zip5(df["customer_zip_code_prefix"])
    return df[df["customer_state"].str.upper().isin(ESTADOS_BRASIL)]


def limpiar_sellers(df):
    df = df.copy()
    df["seller_zip_code_prefix"] = _zip5(df["seller_zip_code_prefix"])
    return df[df["seller_state"].str.upper().isin(ESTADOS_BRASIL)]


def limpiar_reviews(df):
    """Las reseñas son POSTERIORES a la entrega: sólo se usan para el caso de negocio (EDA),
    nunca como variable del modelo."""
    df = df.copy()
    for c in ("review_creation_date", "review_answer_timestamp"):
        df[c] = pd.to_datetime(df[c])
    return df.drop(columns=["review_comment_title", "review_comment_message"])


def limpiar_items(df):
    df = df.copy()
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"])
    return df


def limpiar_products(df):
    """Peso 0 -> faltante. Las medidas faltantes NO se imputan aquí: se imputan dentro del
    Pipeline de cada modelo, con la mediana aprendida sólo con el train de cada fold."""
    df = df.drop(columns=["product_name_lenght", "product_description_lenght", "product_photos_qty"])
    df.loc[df["product_weight_g"] == 0, "product_weight_g"] = np.nan
    # La categoría faltante es una etiqueta, no un valor estimado: se puede rellenar aquí
    df["product_category_name"] = df["product_category_name"].fillna("sin_categoria")
    return df


def limpiar_geolocation(df):
    df = df.copy()
    df["geolocation_zip_code_prefix"] = _zip5(df["geolocation_zip_code_prefix"])
    dentro = df["geolocation_lat"].between(-35.0, 6.0) & df["geolocation_lng"].between(-75.0, -33.0)
    return df[dentro]


FECHAS_ORDERS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]


def limpiar_orders(df):
    """Devuelve (orders_limpio, bitacora). Reglas, en orden:

    1. Se quitan pedidos cancelados / no disponibles.
    2. Se EXCLUYEN pedidos sin fecha de entrega (decisión del equipo).
    3. Censura al final del periodo: se excluyen pedidos cuya fecha estimada es posterior a
       la última entrega registrada (todavía no se podría saber si llegaron tarde).
    """
    df = df.copy()
    for c in FECHAS_ORDERS:
        df[c] = pd.to_datetime(df[c])
    bitacora = {"pedidos_originales": len(df)}

    cancelados = df["order_status"].isin(["canceled", "unavailable"])
    df = df[~cancelados]
    bitacora["cancelados_o_no_disponibles"] = int(cancelados.sum())

    sin_entrega = df["order_delivered_customer_date"].isna()
    df = df[~sin_entrega]
    bitacora["sin_fecha_de_entrega"] = int(sin_entrega.sum())

    fin_datos = df["order_delivered_customer_date"].max()
    censurados = df["order_estimated_delivery_date"] > fin_datos
    df = df[~censurados]
    bitacora["censurados_fecha_estimada_posterior"] = int(censurados.sum())
    bitacora["ultima_entrega_registrada"] = str(fin_datos)
    bitacora["pedidos_limpios"] = len(df)
    return df.copy(), bitacora


def definir_objetivo(orders):
    """delivery_late = 1 si el pedido llegó después de la fecha estimada.

    También calcula `label_conocida_en`: el momento en que la empresa ya SABE el label.
    - Si llegó a tiempo: lo sabe al entregarse.
    - Si llegó tarde: lo sabe en cuanto pasa la fecha estimada sin entrega (fin de ese día).
    => label_conocida_en = min(entrega, estimada + 1 día)
    Se usa para las variables históricas y para la purga del split temporal.
    """
    o = orders.copy()
    entrega = o["order_delivered_customer_date"]
    estimada = o["order_estimated_delivery_date"]
    if config.COMPARAR_POR_DIA:
        o["delivery_late"] = (entrega.dt.normalize() > estimada.dt.normalize()).astype(int)
    else:
        o["delivery_late"] = (entrega > estimada).astype(int)
    o["label_conocida_en"] = np.minimum(entrega, estimada.dt.normalize() + pd.Timedelta(days=1))
    o["dias_retraso_entrega"] = (entrega - estimada).dt.total_seconds() / 86400   # sólo EDA
    return o
