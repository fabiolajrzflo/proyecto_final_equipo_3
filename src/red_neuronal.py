"""Red neuronal en Keras (Módulo 4): MLP para datos tabulares.

Cada entrenamiento (en un fold o el final) hace lo mismo:
  1. Separa la cola más reciente (15 %) de SU conjunto de entrenamiento para el early stopping.
     Así el early stopping nunca ve el fold de validación ni el test.
  2. Ajusta imputación + escalado sólo con la parte de entrenamiento.
  3. Entrena con pesos por clase balanceados y se queda con la mejor época.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
from keras import layers, regularizers
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

CONFIG_BASE = {"capas": (32, 16), "dropout": 0.3, "l2": 1e-3, "lr": 5e-4, "batch": 512}
FRACCION_ES = 0.15


def pesos_balanceados(y):
    n, n1 = len(y), int(np.sum(y))
    return {0: n / (2 * (n - n1)), 1: n / (2 * n1)}


def pesos_por_fila(y):
    cw = pesos_balanceados(y)
    return np.where(np.asarray(y) == 1, cw[1], cw[0]).astype("float32")


def crear_red(n_entradas, capas=(32, 16), dropout=0.3, l2=1e-3, lr=5e-4, **_):
    modelo = keras.Sequential([keras.Input(shape=(n_entradas,))])
    for n in capas:
        modelo.add(layers.Dense(n, activation="relu", kernel_regularizer=regularizers.l2(l2)))
        modelo.add(layers.Dropout(dropout))
    modelo.add(layers.Dense(1, activation="sigmoid"))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy")
    return modelo


class CurvasEnInferencia(keras.callbacks.Callback):
    """Mide train y validación al final de cada época EN MODO INFERENCIA (sin Dropout) y con la
    misma ponderación por clase, para que las curvas sean comparables."""

    def __init__(self, X_tr, y_tr, X_va, y_va):
        super().__init__()
        self.datos = (X_tr, y_tr, pesos_por_fila(y_tr), X_va, y_va, pesos_por_fila(y_va))
        self.registro = {"loss_train": [], "loss_val": [], "auc_train": [], "auc_val": []}

    def on_epoch_end(self, epoch, logs=None):
        X_tr, y_tr, w_tr, X_va, y_va, w_va = self.datos
        p_tr = self.model.predict(X_tr, batch_size=8192, verbose=0).ravel()
        p_va = self.model.predict(X_va, batch_size=8192, verbose=0).ravel()
        self.registro["loss_train"].append(log_loss(y_tr, p_tr, sample_weight=w_tr))
        self.registro["loss_val"].append(log_loss(y_va, p_va, sample_weight=w_va))
        self.registro["auc_train"].append(roc_auc_score(y_tr, p_tr))
        self.registro["auc_val"].append(roc_auc_score(y_va, p_va))


class RedTabular:
    """Envoltura con interfaz tipo scikit-learn: fit(X, y) / predict_proba(X)."""

    def __init__(self, columnas, cfg=None, max_epocas=100, paciencia=8, curvas=False):
        self.columnas = list(columnas)
        self.cfg = {**CONFIG_BASE, **(cfg or {})}
        self.max_epocas, self.paciencia, self.curvas = max_epocas, paciencia, curvas

    def fit(self, X, y):
        keras.utils.set_random_seed(config.RANDOM_STATE)
        X = X[self.columnas]
        y = np.asarray(y).astype("float32")
        corte = int(len(X) * (1 - FRACCION_ES))          # X está en orden cronológico
        self.prep = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scaler", StandardScaler())])
        X_tr = self.prep.fit_transform(X.iloc[:corte]).astype("float32")
        X_es = self.prep.transform(X.iloc[corte:]).astype("float32")
        y_tr, y_es = y[:corte], y[corte:]

        self.red = crear_red(X_tr.shape[1], **self.cfg)
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=self.paciencia,
                                                   min_delta=1e-4, restore_best_weights=True)]
        if self.curvas:
            self.cb_curvas = CurvasEnInferencia(X_tr, y_tr, X_es, y_es)
            callbacks.insert(0, self.cb_curvas)
        self.historia = self.red.fit(
            X_tr, y_tr, validation_data=(X_es, y_es, pesos_por_fila(y_es)),
            epochs=self.max_epocas, batch_size=self.cfg["batch"],
            class_weight=pesos_balanceados(y_tr), callbacks=callbacks, verbose=0,
        )
        self.mejor_epoca = int(np.argmin(self.historia.history["val_loss"])) + 1
        self.epocas_entrenadas = len(self.historia.history["val_loss"])
        return self

    def predict_proba(self, X):
        Xs = self.prep.transform(X[self.columnas]).astype("float32")
        p = self.red.predict(Xs, batch_size=8192, verbose=0).ravel()
        return np.column_stack([1 - p, p])


def ajustar_predecir_red(columnas, cfg=None):
    def _f(X_tr, y_tr, X_va):
        m = RedTabular(columnas, cfg).fit(X_tr, y_tr)
        return m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1]
    return _f


def guardar_red(modelo, carpeta):
    import joblib
    carpeta.mkdir(parents=True, exist_ok=True)
    modelo.red.save(carpeta / "red.keras")
    joblib.dump({"prep": modelo.prep, "columnas": modelo.columnas, "cfg": modelo.cfg,
                 "mejor_epoca": modelo.mejor_epoca}, carpeta / "prep.joblib")


class RedCargada:
    """Red guardada con guardar_red(), con la misma interfaz predict_proba."""

    def __init__(self, carpeta):
        import joblib
        d = joblib.load(carpeta / "prep.joblib")
        self.prep, self.columnas, self.cfg = d["prep"], d["columnas"], d["cfg"]
        self.red = keras.models.load_model(carpeta / "red.keras")

    def predict_proba(self, X):
        Xs = self.prep.transform(X[self.columnas]).astype("float32")
        p = self.red.predict(Xs, batch_size=8192, verbose=0).ravel()
        return np.column_stack([1 - p, p])
