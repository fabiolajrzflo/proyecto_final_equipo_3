"""Etapa 3 del flujo: una fila por pedido, sólo con información disponible AL MOMENTO DE LA COMPRA.

Todas las variables históricas son "point-in-time": para un pedido comprado en el instante t
sólo se usan labels que la empresa ya conocía antes de t (`label_conocida_en < t`).
Por eso no dependen del corte train/test y se pueden crear antes del split sin fuga.
"""
import numpy as np
import pandas as pd

from . import config


# ------------------------------------------------------------------ utilidades
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia sobre la superficie terrestre entre dos puntos (lat/lon en grados)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _eventos(hist, clave):
    """Labels ordenados por el momento en que se conocieron, con acumulados por clave."""
    ev = (
        hist[[clave, "label_conocida_en", "delivery_late"]].dropna()
        .sort_values("label_conocida_en")
        .rename(columns={"label_conocida_en": "t"})
    )
    ev["cum_n"] = ev.groupby(clave).cumcount() + 1
    ev["cum_late"] = ev.groupby(clave)["delivery_late"].cumsum()
    return ev[["t", clave, "cum_n", "cum_late"]]


def _acumulado_antes(consultas, eventos, clave, desfase=None, estricto=True):
    """Acumulados (n, tardíos) de `clave` conocidos antes de t (o de t - desfase)."""
    q = consultas.copy()
    if desfase is not None:
        q["t"] = q["t"] - desfase
    q = q.sort_values("t")
    res = pd.merge_asof(q, eventos, on="t", by=clave, direction="backward",
                        allow_exact_matches=not estricto)
    return res.set_index("order_id")[["cum_n", "cum_late"]].fillna(0)


def tasa_global_previa(hist, consultas):
    """Tasa global de retraso conocida antes de cada compra (prior del suavizado).

    Reemplaza a la `tasa_global_train` del notebook original, que usaba todo el periodo
    de entrenamiento y filtraba información del futuro hacia los primeros folds.
    """
    h = hist.assign(_todo="todo")
    q = consultas.assign(_todo="todo")[["order_id", "_todo", "t"]]
    acc = _acumulado_antes(q, _eventos(h, "_todo"), "_todo")
    return (acc["cum_late"] / acc["cum_n"].replace(0, np.nan)).rename("prior")


def tasa_historica(hist, consultas, clave, nombre, prior, m=config.M_SUAVIZADO):
    """Tasa de retraso suavizada de `clave` (vendedor, ruta...) con todo su historial previo."""
    acc = _acumulado_antes(consultas[["order_id", clave, "t"]], _eventos(hist, clave), clave)
    p = prior.reindex(acc.index)
    return pd.DataFrame({
        f"{nombre}_late_rate": (acc["cum_late"] + m * p) / (acc["cum_n"] + m),
        f"{nombre}_n_prev": acc["cum_n"],
    })


def tasa_reciente(hist, consultas, clave, nombre, prior, dias=config.DIAS_VENTANA, m=config.M_SUAVIZADO):
    """Tasa de retraso de los labels conocidos en los `dias` previos a la compra."""
    if clave is None:
        hist, consultas, clave = hist.assign(_todo="todo"), consultas.assign(_todo="todo"), "_todo"
    ev = _eventos(hist, clave)
    q = consultas[["order_id", clave, "t"]]
    hasta = _acumulado_antes(q, ev, clave)
    desde = _acumulado_antes(q, ev, clave, desfase=pd.Timedelta(days=dias), estricto=False).reindex(hasta.index)
    n = hasta["cum_n"] - desde["cum_n"]
    tardios = hasta["cum_late"] - desde["cum_late"]
    p = prior.reindex(hasta.index)
    return pd.DataFrame({
        f"{nombre}_late_{dias}d": (tardios + m * p) / (n + m),
        f"{nombre}_n_{dias}d": n,
    })


# ------------------------------------------------------------------ dataset por pedido
def construir_dataset(t, orders_raw):
    """t: dict de tablas limpias (01_limpieza). orders_raw: orders originales (todas las
    compras, incluso canceladas) para medir la carga de pedidos de los últimos 7 días."""
    orders = t["orders"]

    # ---- Pedido, fechas y cliente
    df = orders[["order_id", "customer_id", "order_purchase_timestamp", "order_estimated_delivery_date",
                 "label_conocida_en", "delivery_late"]].copy()
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek.astype(str)
    df["franja_horaria"] = pd.cut(df["order_purchase_timestamp"].dt.hour, bins=[-1, 5, 11, 17, 23],
                                  labels=["madrugada", "manana", "tarde", "noche"]).astype(str)
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df = df.drop(columns=["order_estimated_delivery_date"])
    df = df.merge(t["customers"][["customer_id", "customer_state", "customer_zip_code_prefix"]],
                  on="customer_id", how="left", validate="one_to_one")

    # ---- Ítems, productos y vendedor
    items = (
        t["items"]
        .merge(t["products"], on="product_id", how="left", validate="many_to_one")
        .merge(t["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left", validate="many_to_one")
        .merge(orders[["order_id", "order_purchase_timestamp"]], on="order_id", how="inner", validate="many_to_one")
        .sort_values(["order_id", "order_item_id"])
    )
    items["dias_limite_envio"] = (
        items["shipping_limit_date"] - items["order_purchase_timestamp"]).dt.total_seconds() / 86400
    items_agg = items.groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_products=("product_id", "nunique"),
        total_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_product_weight=("product_weight_g", "mean"),
        max_product_weight=("product_weight_g", "max"),
        avg_product_length=("product_length_cm", "mean"),
        avg_product_height=("product_height_cm", "mean"),
        avg_product_width=("product_width_cm", "mean"),
        unique_product_categories=("product_category_name", "nunique"),
        avg_dias_limite_envio=("dias_limite_envio", "mean"),
        min_dias_limite_envio=("dias_limite_envio", "min"),
        main_seller_id=("seller_id", "first"),
        seller_state=("seller_state", "first"),
        n_seller_states=("seller_state", "nunique"),
        main_category=("product_category_name", "first"),
    ).reset_index()
    df = df.merge(items_agg, on="order_id", how="left", validate="one_to_one")

    # ---- Pagos
    pay = t["payments"]
    pay_agg = pay.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_types=("payment_type", "nunique"),
    )
    pay_agg["main_payment_type"] = pay.groupby("order_id")["payment_type"].agg(lambda x: x.mode().iloc[0])
    df = df.merge(pay_agg.reset_index(), on="order_id", how="left", validate="one_to_one")

    # ---- Distancia vendedor-cliente (Haversine sobre el centroide de cada código postal)
    geo = t["geolocation"].groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    cli = t["customers"].set_index("customer_id")["customer_zip_code_prefix"].map(geo["geolocation_lat"]).rename("c_lat").to_frame()
    cli["c_lng"] = t["customers"].set_index("customer_id")["customer_zip_code_prefix"].map(geo["geolocation_lng"])
    sel = t["sellers"].set_index("seller_id")["seller_zip_code_prefix"].map(geo["geolocation_lat"]).rename("s_lat").to_frame()
    sel["s_lng"] = t["sellers"].set_index("seller_id")["seller_zip_code_prefix"].map(geo["geolocation_lng"])
    dist = (items[["order_id", "seller_id"]]
            .merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
            .join(cli, on="customer_id").join(sel, on="seller_id"))
    dist["distance_km"] = haversine_km(dist["c_lat"], dist["c_lng"], dist["s_lat"], dist["s_lng"])
    dist_agg = dist.groupby("order_id")["distance_km"].agg(avg_distance_km="mean", max_distance_km="max")
    df = df.merge(dist_agg.reset_index(), on="order_id", how="left", validate="one_to_one")

    # ---- Variables de ruta
    df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
    df["ruta"] = df["seller_state"].fillna("NA") + "_" + df["customer_state"].fillna("NA")
    df["freight_ratio"] = df["total_freight"] / (df["total_price"] + 1e-6)

    # ---- Pedidos sin ítems o sin pago: no hay información para predecir
    df = df[df["total_items"].notna() & df["total_payment"].notna()].copy()
    df["payment_types"] = df["payment_types"].astype("int64")
    df["max_installments"] = df["max_installments"].astype("int64")

    # ---- Variables históricas (point-in-time)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    consultas = df[["order_id", "main_seller_id", "ruta", "customer_state"]].assign(t=df["order_purchase_timestamp"])
    hist = df[["order_id", "main_seller_id", "ruta", "customer_state", "label_conocida_en", "delivery_late"]]

    prior = tasa_global_previa(hist, consultas)
    df = (df
          .join(tasa_historica(hist, consultas, "main_seller_id", "seller", prior), on="order_id")
          .join(tasa_historica(hist, consultas, "ruta", "ruta", prior), on="order_id")
          .join(tasa_reciente(hist, consultas, None, "global", prior), on="order_id")
          .join(tasa_reciente(hist, consultas, "customer_state", "estado", prior)
                .drop(columns=f"estado_n_{config.DIAS_VENTANA}d"), on="order_id"))

    # ---- Carga de pedidos: compras (de cualquier estado) en los 7 días previos
    compras = np.sort(pd.to_datetime(orders_raw["order_purchase_timestamp"]).to_numpy())
    t_c = df["order_purchase_timestamp"].to_numpy()
    df["pedidos_ult_7d"] = (np.searchsorted(compras, t_c, side="left")
                            - np.searchsorted(compras, t_c - np.timedelta64(7, "D"), side="left"))

    assert df["order_id"].is_unique
    assert df["order_purchase_timestamp"].is_monotonic_increasing
    return df
