"""Configuración central del proyecto: rutas, semillas y decisiones del equipo.

Todos los notebooks importan de aquí, así una decisión se cambia en un solo lugar.
"""
from pathlib import Path

# ---------------------------------------------------------------- Rutas
RAIZ = Path(__file__).resolve().parents[1]
DATOS_RAW = RAIZ / "datos"                     # CSV originales de Olist (no se modifican)
DATOS_INTERIM = RAIZ / "datos" / "interim"     # tablas limpias (01_limpieza)
DATOS_PROCESSED = RAIZ / "datos" / "processed" # dataset por pedido y splits (03, 04)
MODELOS = RAIZ / "modelos"                     # artefactos de cada modelo (05x)
REPORTES = RAIZ / "reportes"
FIGURAS = REPORTES / "figuras"
METRICAS = REPORTES / "metricas"

for _d in (DATOS_INTERIM, DATOS_PROCESSED, MODELOS, FIGURAS, METRICAS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Reproducibilidad
RANDOM_STATE = 42

# ---------------------------------------------------------------- Variable objetivo
# True: un pedido entregado el MISMO DÍA estimado cuenta como "a tiempo".
COMPARAR_POR_DIA = True

# ---------------------------------------------------------------- Periodo de estudio
# 2016 tiene muy pocos pedidos y un hueco en nov-dic: se usa sólo como historial
# para las variables históricas, no como filas del modelo.
FECHA_INICIO_MODELO = "2017-01-01"

# ---------------------------------------------------------------- Split temporal
# El test es el bloque más reciente. Se fija por fecha (no por % de filas) para que
# sea fácil de explicar. Ver 04_preparacion_dataset.
FECHA_CORTE_TEST = "2018-05-01"
N_FOLDS = 5          # K-Fold temporal (ventana creciente) dentro del train

# ---------------------------------------------------------------- Features
M_SUAVIZADO = 20     # pedidos "imaginarios" para suavizar tasas históricas
DIAS_VENTANA = 30    # ventana de las variables de carga reciente
MAX_CATEGORIAS = 20  # one-hot: categorías más frecuentes (en train); el resto -> "infrequent"

# ---------------------------------------------------------------- Umbral de decisión
# "f1": el umbral que maximiza el F1 de la clase 1 en las predicciones fuera de fold.
CRITERIO_UMBRAL = "f1"

# ---------------------------------------------------------------- Modelos
# Orden de simplicidad: en un empate de PR-AUC (< TOLERANCIA_EMPATE) gana el más simple.
MODELOS_CLASICOS = ["Regresión Logística", "Random Forest", "XGBoost"]
NOMBRE_RED = "Red Neuronal"
TOLERANCIA_EMPATE = 0.005
CLAVES_MODELO = {
    "Regresión Logística": "logistica",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Red Neuronal": "red_neuronal",
}
