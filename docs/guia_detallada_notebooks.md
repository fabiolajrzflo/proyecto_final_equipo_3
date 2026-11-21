# Guía detallada del código · Proyecto Olist (entregas tardías)

> **Para quién es:** para entender (y poder explicar) cada celda de los 9 notebooks, como si se lo contaras a alguien de secundaria.
> **Cómo usarla:** abre el notebook al lado de esta guía. Los números de celda coinciden con el orden en el notebook (Celda 1 = la primera de arriba).
> Las celdas de **texto** (markdown) sólo explican; las de **código** hacen el trabajo.

---

## Índice

0. [Antes de empezar: vocabulario básico](#0-antes-de-empezar-vocabulario-básico)
1. [La celda de preparación (aparece en todos los notebooks)](#1-la-celda-de-preparación-aparece-en-todos-los-notebooks)
2. [Notebook 01 · Limpieza](#notebook-01--limpieza-de-datos)
3. [Notebook 02 · Exploración](#notebook-02--exploración-de-datos-eda)
4. [Notebook 03 · Creación de variables](#notebook-03--creación-de-variables-feature-engineering)
5. [Notebook 04 · Preparación del dataset](#notebook-04--preparación-del-dataset-split-temporal-purga-y-folds)
6. [Notebooks 05a, 05b, 05c · Modelos clásicos](#notebooks-05a-05b-05c--modelos-clásicos)
7. [Notebook 05d · Red neuronal](#notebook-05d--red-neuronal-en-keras)
8. [Notebook 06 · Selección y prueba final](#notebook-06--selección-del-mejor-modelo-y-prueba-final)
9. [Anexo A · Las funciones de `src/` por dentro](#anexo-a--las-funciones-de-src-por-dentro)
10. [Anexo B · Glosario de funciones de librerías](#anexo-b--glosario-de-funciones-de-librerías)

---

## 0. Antes de empezar: vocabulario básico

| Palabra | Qué significa (en simple) |
|---|---|
| **Python** | El lenguaje en el que está escrito todo. |
| **Librería** | Una caja de herramientas que alguien ya programó. Ej.: `pandas` (tablas), `numpy` (números), `matplotlib` (gráficas), `scikit-learn` (modelos), `xgboost`, `keras` (redes neuronales). |
| **`import`** | "Saca esta caja de herramientas para usarla". `import pandas as pd` = usa pandas y llámalo `pd` para escribir menos. |
| **Variable** | Una cajita con nombre donde guardas algo. `x = 5` guarda el 5 en la cajita `x`. |
| **Función** | Una receta con nombre: le das ingredientes (argumentos) y te regresa un resultado. `len([1,2,3])` regresa `3`. |
| **DataFrame** | Una **tabla** como de Excel: filas y columnas. Es lo que usa pandas. |
| **Serie** | Una sola columna de una tabla. |
| **Diccionario (`dict`)** | Una lista de pares "nombre → valor". `{"edad": 15, "nombre": "Ana"}`. |
| **Lista** | Varios elementos en orden: `[1, 2, 3]`. |
| **CSV** | Archivo de texto con una tabla separada por comas. Así vienen los datos de Olist. |
| **Parquet** | Otro formato de tabla, comprimido y mucho más rápido de leer que CSV. Lo usamos para guardar resultados intermedios. |
| **JSON** | Archivo de texto con datos en formato "nombre: valor". Lo usamos para guardar resúmenes y números para la presentación. |
| **`src/`** | Carpeta con nuestro propio código reutilizable. Los notebooks lo importan para no copiar y pegar las mismas funciones. |
| **Label / variable objetivo** | Lo que queremos predecir: `delivery_late` (1 = llegó tarde, 0 = a tiempo). |
| **Feature / variable** | Un dato que el modelo usa para predecir (distancia, precio, historial del vendedor…). |
| **Fuga de información (data leakage)** | Hacer trampa sin querer: que el modelo vea datos que en la vida real todavía no existirían. Es como estudiar con las respuestas del examen. |

**La idea de todo el proyecto en una frase:** con lo que Olist sabe *el día que alguien compra*, queremos estimar la probabilidad de que ese pedido llegue tarde.

---

## 1. La celda de preparación (aparece en todos los notebooks)

Es la **Celda 2** de cada notebook. Siempre hace lo mismo: prepara el "escritorio de trabajo".

```python
import sys, json, time, warnings
from pathlib import Path

RAIZ = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from src import config, viz
viz.aplicar_estilo()
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)
```

| Línea | Qué hace |
|---|---|
| `import sys, json, time, warnings` | Herramientas básicas de Python: `sys` (configurar Python), `json` (leer/escribir archivos JSON), `time` (medir cuánto tarda algo), `warnings` (avisos). |
| `from pathlib import Path` | Herramienta para manejar rutas de carpetas y archivos. |
| `RAIZ = ...` | Averigua dónde está la carpeta principal del proyecto. Si el notebook se abre desde `notebooks/`, la raíz es la carpeta de arriba (`.parent`). |
| `sys.path.insert(0, str(RAIZ))` | Le dice a Python: "también busca código en la carpeta del proyecto". Así puede encontrar `src/`. |
| `warnings.filterwarnings("ignore")` | Oculta avisos que no son errores, para que la salida se vea limpia. |
| `import numpy as np` | Números y operaciones matemáticas rápidas. |
| `import pandas as pd` | Tablas (DataFrames). |
| `import matplotlib.pyplot as plt` | Gráficas. |
| `from IPython.display import display, Markdown` | `display()` muestra tablas bonitas en el notebook; `Markdown()` muestra texto con formato. |
| `from src import config, viz` | Importa dos archivos nuestros: `config` (rutas y decisiones del equipo) y `viz` (estilo de gráficas). |
| `viz.aplicar_estilo()` | Pone a todas las gráficas los mismos colores, letras y estilo. |
| `pd.set_option(...)` | Que pandas muestre hasta 100 columnas y use más ancho de pantalla al imprimir tablas. |

**¿Qué hay en `config`?** Es el "tablero de control" del proyecto:
- Rutas: dónde están los datos (`DATOS_RAW`), dónde guardar tablas limpias (`DATOS_INTERIM`), procesadas (`DATOS_PROCESSED`), modelos (`MODELOS`), figuras (`FIGURAS`) y métricas (`METRICAS`).
- `RANDOM_STATE = 42`: la "semilla" del azar. Hace que cada vez que corras el código salgan los mismos resultados.
- `COMPARAR_POR_DIA = True`: un pedido entregado el mismo día prometido cuenta como a tiempo.
- `FECHA_INICIO_MODELO = "2017-01-01"` y `FECHA_CORTE_TEST = "2018-05-01"`.
- `N_FOLDS = 5`, `M_SUAVIZADO = 20`, `DIAS_VENTANA = 30`, `MAX_CATEGORIAS = 20`, `TOLERANCIA_EMPATE = 0.005`.

Si el equipo quiere cambiar una decisión (por ejemplo, la fecha del test), se cambia **sólo ahí** y todos los notebooks la usan.

---

## Notebook 01 · Limpieza de datos

**Objetivo:** cargar las tablas originales de Olist, arreglar errores, definir qué es "llegar tarde" y guardar tablas limpias.
**Entra:** los CSV de `datos/` · **Sale:** tablas `.parquet` en `datos/interim/` y `reportes/metricas/limpieza.json`.

### Celda 1 (texto)
Título, el flujo del proyecto (el diagrama del profesor) y las reglas de esta etapa: no rellenar huecos con datos de todo el periodo, no borrar outliers y documentar cada fila eliminada.

### Celda 2 (código) · Preparación
La celda de preparación de la sección 1, más `from src import limpieza as L`: trae nuestras funciones de limpieza con el apodo `L`.

### Celda 3 (código) · Cargar las tablas
```python
raw = L.cargar_tablas()
display(pd.DataFrame({k: {"filas": v.shape[0], "columnas": v.shape[1]} for k, v in raw.items()}).T)
```
- `L.cargar_tablas()` lee los 8 archivos CSV (clientes, vendedores, reseñas, ítems, productos, geolocalización, pedidos, pagos) y los guarda en un **diccionario** llamado `raw` ("crudo"). Ej.: `raw["orders"]` es la tabla de pedidos.
- La segunda línea arma una tablita con cuántas filas y columnas tiene cada tabla. `v.shape` da (filas, columnas). `.T` voltea la tabla (filas ↔ columnas) para que se lea mejor.

**Para qué:** ver qué datos tenemos y de qué tamaño.

### Celdas 4–5 · Customers (clientes)
```python
c = raw["customers"]
print("customer_id duplicados:", c.duplicated("customer_id").sum())
print("Clientes únicos con varias compras:", c.duplicated("customer_unique_id").sum())
customers = L.limpiar_customers(c)
print("Eliminados por estado inválido:", len(c) - len(customers))
display(customers["customer_state"].value_counts(normalize=True).mul(100).round(2).head(10).to_frame("% de clientes"))
```
- `duplicated("customer_id").sum()` cuenta filas con un ID repetido. Da 0: cada pedido tiene su propio `customer_id`.
- `customer_unique_id` sí se repite: es la **persona real**, y una persona puede comprar varias veces.
- `L.limpiar_customers(c)` hace dos cosas:
  1. Convierte el código postal a texto de 5 dígitos con `zfill(5)` (`"1234"` → `"01234"`), para no perder los ceros de la izquierda.
  2. Se queda sólo con clientes cuyo estado está en la lista oficial de los 27 estados de Brasil.
- `len(c) - len(customers)` = cuántos se eliminaron.
- `value_counts(normalize=True)` cuenta cuántos clientes hay por estado, como proporción. `.mul(100)` lo pasa a porcentaje, `.round(2)` redondea a 2 decimales, `.head(10)` muestra los 10 primeros.

**Para qué:** asegurar que las claves y los estados son válidos, y ver de dónde son los clientes (SP domina).

### Celdas 6–7 · Sellers (vendedores)
Lo mismo que con clientes, pero para vendedores: revisa IDs duplicados, arregla el código postal y quita estados inválidos (`L.limpiar_sellers`). Muestra el top 10 de estados de vendedores.

### Celdas 8–9 · Order Reviews (reseñas)
```python
reviews = L.limpiar_reviews(raw["reviews"])
display(reviews["review_score"].value_counts(normalize=True).sort_index().mul(100).round(2).to_frame("% de reseñas"))
```
- `limpiar_reviews` convierte las fechas de texto a fechas reales (`pd.to_datetime`) y borra las columnas del título y del comentario (texto libre que no usamos).
- Luego muestra qué porcentaje de reseñas tiene 1, 2, 3, 4 y 5 estrellas. `sort_index()` las ordena de 1 a 5.

**Importante:** las reseñas se escriben **después** de la entrega, así que **nunca** entran al modelo (sería trampa). Sólo sirven para demostrar que un retraso hace enojar al cliente.

### Celdas 10–11 · Order Items (productos dentro de cada pedido)
```python
items = L.limpiar_items(raw["items"])
display(items[["price", "freight_value"]].describe().round(2))
print("Pedidos con algún ítem con flete 0 (envío gratis):", items.loc[items["freight_value"] == 0, "order_id"].nunique())
```
- `limpiar_items` convierte `shipping_limit_date` (fecha límite para que el vendedor entregue el paquete a la paquetería) a fecha. Esta fecha **sí se conoce al comprar**, por eso luego la usamos.
- `.describe()` da un resumen estadístico: cuántos hay, promedio, desviación, mínimo, cuartiles y máximo del precio y del flete.
- `items.loc[condición, "order_id"].nunique()` = cuántos pedidos **distintos** tienen algún producto con flete 0.

### Celdas 12–13 · Products (productos)
```python
print("Productos con peso 0:", (raw["products"]["product_weight_g"] == 0).sum())
products = L.limpiar_products(raw["products"])
print(products[L.COLS_DIM].isna().sum())
```
- Cuenta productos con peso 0 (imposible en la realidad).
- `limpiar_products`:
  1. Borra 3 columnas que no usamos (largo del nombre, de la descripción y número de fotos).
  2. Cambia el peso 0 por "faltante" (`NaN`, "no es un número").
  3. Rellena la categoría vacía con `"sin_categoria"`.
- `.isna().sum()` cuenta cuántos faltantes quedan en peso y medidas.

**¿Por qué no rellenamos los faltantes de peso aquí?** Porque para rellenarlos usaríamos la mediana de *todos* los productos, incluidos los que se venden en el periodo de prueba (el "futuro"). Eso sería una pequeña trampa. Se rellenan después, **dentro de cada modelo**, usando sólo los datos de entrenamiento.

### Celdas 14–15 · Payments (pagos)
Muestra un resumen de las cuotas (`payment_installments`) y del monto pagado, y qué porcentaje de pagos es con tarjeta, boleto, voucher, etc. Esta tabla no se modifica.

### Celdas 16–17 · Geolocation (coordenadas)
```python
geolocation = L.limpiar_geolocation(raw["geolocation"])
```
`limpiar_geolocation` arregla el código postal a 5 dígitos y se queda con las coordenadas que caen **dentro de Brasil** (latitud entre −35 y 6, longitud entre −75 y −33). Elimina puntos que por error caían en otros países u océanos.

### Celda 18 (texto) · Orders y variable objetivo
Explica tres cosas:
1. **El label:** `delivery_late = 1` si el pedido llegó un día posterior a la fecha prometida. Comparamos **por día** porque la fecha prometida siempre viene a las 00:00.
2. **Las reglas de limpieza de pedidos**, en orden: quitar cancelados, excluir los que no tienen fecha de entrega y la "censura" del final del periodo.
3. **¿Cuándo se conoce el label?** `label_conocida_en = min(entrega, fecha estimada + 1 día)`. Si llegó a tiempo, lo sabes cuando se entrega. Si llegó tarde, lo sabes en cuanto pasa la fecha prometida sin que llegue.

### Celda 19 (código) · Diagnóstico de los pedidos
```python
o = raw["orders"].copy()
for col in L.FECHAS_ORDERS:
    o[col] = pd.to_datetime(o[col])
display(o["order_status"].value_counts(normalize=True).mul(100).round(2).to_frame("% de pedidos"))
est = o["order_estimated_delivery_date"]
print("¿Todas las fechas estimadas son a las 00:00?", bool((est == est.dt.normalize()).all()))
no_cancelados = o[~o["order_status"].isin(["canceled", "unavailable"])]
diag = (no_cancelados.assign(mes=..., sin_entrega=...).groupby("mes")["sin_entrega"].agg(pedidos="size", sin_entrega="sum"))
display(diag.tail(8))
```
- `.copy()` hace una copia para no modificar la tabla original.
- El `for` recorre las 5 columnas de fecha y las convierte de texto a fecha.
- Muestra el porcentaje de pedidos por estado (`delivered`, `shipped`, `canceled`…).
- `est.dt.normalize()` pone la hora en 00:00. Si es igual a la original en **todas** las filas (`.all()`), entonces todas las fechas estimadas ya venían a las 00:00. Esto justifica comparar por día.
- `~` significa "NO". `~o["order_status"].isin([...])` = pedidos que **no** están cancelados.
- `assign` agrega columnas temporales: el mes de compra y si le falta la fecha de entrega. `groupby("mes")` agrupa por mes (como una tabla dinámica) y `agg` cuenta pedidos y faltantes.

**Para qué:** ver que los pedidos sin fecha de entrega se concentran al final, y decidir qué hacer con ellos (el equipo decidió excluirlos).

### Celda 20 (código) · Limpiar pedidos y crear el label
```python
orders, bitacora = L.limpiar_orders(raw["orders"])
orders = L.definir_objetivo(orders)
display(pd.Series(bitacora, name="valor").to_frame())
late_hora = (entrega > estimada).astype(int)
print(... late_hora ..., ... orders['delivery_late'] ...)
```
- `limpiar_orders` aplica las 3 reglas y regresa **dos cosas**: la tabla limpia y una "bitácora" (diccionario) con cuántos pedidos quitó en cada paso. De 99,441 pedidos quedan 96,468.
- `definir_objetivo` crea tres columnas:
  - `delivery_late` (0/1),
  - `label_conocida_en` (cuándo se sabe el resultado),
  - `dias_retraso_entrega` (días entre la entrega real y la prometida; sólo para graficar).
- Al final compara cuántos tardíos hay si se compara con hora vs. sólo por día. `.astype(int)` convierte True/False en 1/0.

### Celda 21 (texto) · Validaciones y guardado

### Celda 22 (código) · Validar y guardar
```python
assert orders["order_id"].is_unique
assert orders["delivery_late"].isin([0, 1]).all()
...
for nombre, tabla in tablas.items():
    tabla.to_parquet(config.DATOS_INTERIM / f"{nombre}.parquet", index=False)
(config.METRICAS / "limpieza.json").write_text(json.dumps(bitacora, indent=2))
```
- `assert` = "asegúrate de que esto sea verdad; si no, detente con error". Son **pruebas automáticas**: que cada pedido aparezca una sola vez, que el label sólo valga 0 o 1, que todos tengan fecha de entrega y que, en prácticamente todos los pedidos (> 99.9 %), la fecha en que se conoce el label no sea anterior a la compra (hay unos pocos registros con fechas erróneas en los datos originales).
- El `for` guarda cada tabla limpia como `.parquet`. `f"{nombre}.parquet"` arma el nombre del archivo (ej. `orders.parquet`).
- `json.dumps(bitacora)` convierte la bitácora a texto JSON y la guarda para usarla en la presentación y el artículo.

---

## Notebook 02 · Exploración de datos (EDA)

**Objetivo:** entender los datos y encontrar historias: el costo de llegar tarde, las oleadas de retraso, la geografía.
**Entra:** `datos/interim/` · **Sale:** figuras en `reportes/figuras/` y `reportes/metricas/eda.json`.

**Regla anti-trampa:** la descripción general puede usar todo el periodo, pero cualquier análisis **contra el label** que pueda influir en decisiones del modelo se hace sólo con datos anteriores al test.

### Celda 1 (texto)
Título, flujo y la regla anti-trampa.

### Celda 2 (código) · Preparación
Igual que siempre.

### Celda 3 (código) · Cargar y filtrar
```python
t = {n: pd.read_parquet(config.DATOS_INTERIM / f"{n}.parquet") for n in ["orders", "customers", ...]}
orders = t["orders"]
orders = orders[orders["order_purchase_timestamp"] >= config.FECHA_INICIO_MODELO].copy()
corte = pd.Timestamp(config.FECHA_CORTE_TEST)
es_train = orders["order_purchase_timestamp"] < corte
eda = {"pedidos": int(len(orders)), "tasa_tardios": float(orders["delivery_late"].mean())}
```
- La primera línea es un **diccionario por comprensión**: lee cada tabla limpia y la guarda con su nombre. Es una forma corta de escribir un `for`.
- Se queda con pedidos desde enero 2017 (2016 tiene muy pocos pedidos).
- `corte` es la fecha de inicio del test (1 de mayo de 2018).
- `es_train` es una columna de verdadero/falso: True si el pedido es anterior al test.
- `eda` es un diccionario donde vamos guardando números importantes para la presentación. `.mean()` de una columna de 0 y 1 = proporción de unos = **tasa de tardíos** (6.8 %).

### Celdas 4–5 · El costo de llegar tarde (caso de negocio)
```python
rev = t["reviews"].groupby("order_id")["review_score"].mean().rename("score")
o_rev = orders.join(rev, on="order_id").dropna(subset=["score"])
resumen_rev = o_rev.groupby("delivery_late")["score"].agg(promedio="mean", pct_1_2_estrellas=..., pct_5_estrellas=...)
dist = (o_rev.assign(estrellas=o_rev["score"].round().astype(int)).groupby("delivery_late")["estrellas"]
        .value_counts(normalize=True).mul(100).unstack(0).sort_index())
fig, ax = plt.subplots(...); ax.bar(...); ...
viz.guardar(fig, "eda_reviews")
eda["reviews"] = {...}
```
- Algunos pedidos tienen más de una reseña: `groupby("order_id")...mean()` calcula la calificación promedio por pedido.
- `join` pega esa calificación a cada pedido (como BUSCARV en Excel). `dropna` quita pedidos sin reseña.
- `groupby("delivery_late")` separa en dos grupos (a tiempo / tarde) y `agg` calcula, para cada grupo, el promedio de estrellas, el % con 1–2 estrellas y el % con 5 estrellas. Las `lambda` son mini-funciones de una línea: `lambda s: (s <= 2).mean() * 100` = porcentaje de calificaciones menores o iguales a 2.
- `dist` calcula qué % de cada grupo dio 1, 2, 3, 4 o 5 estrellas. `unstack(0)` acomoda la tabla con una columna por grupo.
- `plt.subplots` crea el lienzo de la gráfica y `ax.bar` dibuja barras: azul para "a tiempo" y rojo para "tardío", una al lado de la otra (por eso `x - 0.2` y `x + 0.2`).
- `viz.guardar` guarda la imagen en `reportes/figuras/`.

**Resultado clave:** tardío ≈ 2.3★ vs. a tiempo ≈ 4.3★. El 62 % de los tardíos da 1–2 estrellas.

### Celdas 6–7 · Los retrasos llegan en oleadas
```python
mensual = (orders.set_index("order_purchase_timestamp")["delivery_late"]
           .resample("MS").agg(["size", "mean"]).rename(...))
fig, axes = plt.subplots(2, 1, ...)
axes[0].bar(...)    # pedidos por mes
axes[1].plot(...)   # % de tardíos por mes
ax.axvline(corte, ...)
```
- `set_index(fecha)` pone la fecha como índice para poder agrupar por tiempo.
- `.resample("MS")` agrupa por mes ("Month Start"). `size` cuenta pedidos y `mean` da la tasa de tardíos del mes.
- Dos gráficas, una arriba de otra: barras con el volumen y línea con el % de tardíos.
- `axvline` dibuja una línea vertical en la fecha de inicio del test.

**Para qué:** muestra que la tasa sube muchísimo en algunos periodos (Black Friday 2017, feb–mar 2018). Eso justifica: (a) dividir los datos por tiempo, (b) crear variables de "carga reciente" y (c) no usar el mes como variable.

### Celdas 8–9 · La fecha prometida ya trae "colchón"
```python
d = orders["dias_retraso_entrega"]
ax.hist(d.clip(-50, 30), bins=80, ...)
ax.hist(d[d > 0].clip(-50, 30), bins=..., color=viz.TARDIO)
ax.set_title(f"Mediana: el pedido llega {abs(d.median()):.0f} días antes de lo prometido")
```
- Histograma = gráfica de "cuántos pedidos hay en cada rango de días".
- `.clip(-50, 30)` recorta valores extremos para que la gráfica se lea (todo lo menor a −50 se pone en −50).
- Se pinta de rojo la parte > 0 (tardíos).
- La **mediana** (el valor de en medio) dice que el pedido típico llega unos 12 días **antes** de lo prometido.

**Para qué:** explicar por qué el problema es difícil: Olist ya promete fechas con margen, así que los retrasos que quedan son "sorpresas".

### Celdas 10–11 · Outliers (método IQR)
```python
def pct_outliers(serie):
    s = serie.dropna(); q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    return pd.Series({"% bajo límite": (s < q1 - 1.5 * iqr).mean() * 100, "% sobre límite": (s > q3 + 1.5 * iqr).mean() * 100})
```
- `def` crea una función propia.
- Q1 es el valor que deja al 25 % de los datos abajo y Q3 el que deja al 75 %. IQR = Q3 − Q1 (el "ancho" de la mitad central).
- Un **outlier** (valor atípico) es lo que queda muy lejos: menos de Q1 − 1.5·IQR o más de Q3 + 1.5·IQR.
- La función regresa el % de outliers abajo y arriba, y se aplica a varias columnas para armar una tabla.

**Decisión:** **no** los borramos, porque los pedidos extremos (muy pesados o lejanos) son justamente los que más se retrasan.

### Celdas 12–13 · Geografía del retraso (sólo train)
```python
o_train = orders[es_train].merge(t["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
por_estado = o_train.groupby("customer_state")["delivery_late"].agg(pedidos="size", tasa="mean")
por_estado = por_estado[por_estado["pedidos"] >= 300].sort_values("tasa")
ax.barh(...)
```
- Sólo pedidos de entrenamiento (`es_train`), porque aquí sí comparamos contra el label.
- `merge` une pedidos con clientes para saber el estado de cada cliente.
- Por estado: cuántos pedidos y qué tasa de retraso. Se quedan los estados con al menos 300 pedidos (con pocos pedidos el porcentaje no es confiable).
- `barh` = barras horizontales. La línea punteada es el promedio.

### Celdas 14–15 · Precio y flete
Dos diagramas de caja (boxplots) en escala logarítmica (`set_xscale("log")`), que sirve para ver datos que van de muy pequeños a muy grandes. Al final guarda el diccionario `eda` en `eda.json`.

---

## Notebook 03 · Creación de variables (feature engineering)

**Objetivo:** construir **una fila por pedido** con todas las variables que el modelo usará, **sólo con información conocida al momento de la compra**.
**Entra:** `datos/interim/` · **Sale:** `datos/processed/dataset_pedidos.parquet`.

### Celda 1 (texto)
Explica el principio *point-in-time* ("en el punto del tiempo"): para un pedido comprado el día *t*, sólo se cuentan resultados que ya se conocían antes de *t*. También explica la corrección respecto a la versión anterior: antes, el "valor de referencia" del suavizado se calculaba con todo el periodo de entrenamiento, lo cual metía información del futuro en los primeros folds.

### Celda 2 (código) · Preparación
Igual que siempre, más `from src import features as F`.

### Celda 3 (código) · Construir el dataset
```python
t = {n: pd.read_parquet(...) for n in ["orders", "customers", "sellers", "items", "products", "payments", "geolocation"]}
orders_raw = pd.read_csv(config.DATOS_RAW / "olist_orders_dataset.csv", usecols=["order_purchase_timestamp"])
df_completo = F.construir_dataset(t, orders_raw)
df = df_completo[df_completo["order_purchase_timestamp"] >= config.FECHA_INICIO_MODELO].reset_index(drop=True)
```
- Carga las tablas limpias.
- `orders_raw` lee **sólo** la columna de fecha de compra de **todos** los pedidos (incluidos cancelados). Sirve para medir cuántos pedidos entraron en los últimos 7 días: eso la empresa sí lo sabe al momento de la compra.
- `F.construir_dataset` es la función más importante del proyecto (se explica completa en el [Anexo A](#featurespy)). Hace, en orden:
  1. **Pedido y cliente:** día de la semana, franja horaria (madrugada/mañana/tarde/noche), días prometidos de entrega y estado del cliente.
  2. **Ítems y productos:** junta los productos de cada pedido y calcula totales y promedios (número de ítems, precio total, flete total, peso promedio, medidas, plazo del vendedor para despachar, vendedor y categoría principal).
  3. **Pagos:** total pagado, máximo de cuotas, cuántos tipos de pago y el tipo principal.
  4. **Distancia:** km entre vendedor y cliente con la fórmula de Haversine (distancia sobre una esfera).
  5. **Ruta:** si vendedor y cliente están en el mismo estado, la ruta "SP_RJ" y la proporción flete/precio.
  6. **Historial (point-in-time):** tasa de retraso previa del vendedor y de la ruta.
  7. **Carga reciente:** tasa de retraso del sistema en los últimos 30 días (global y hacia el estado del cliente) y pedidos de los últimos 7 días.
- Se calcula con todo (incluido 2016, que sirve de "historia") y luego se filtra desde 2017. `reset_index(drop=True)` renumera las filas desde 0.

### Celda 4 (texto) · Diccionario de variables
Una tabla con los 8 grupos de variables y qué captura cada uno. Muy útil para explicarle al jurado.

### Celda 5 (texto) · Verificación de no-fuga

### Celda 6 (código) · Prueba de que no hay trampa
```python
muestra = df.sample(300, random_state=config.RANDOM_STATE)
errores = 0
for _, fila in muestra.iterrows():
    mismo = df_completo["main_seller_id"] == fila["main_seller_id"]
    conocidos = (df_completo["label_conocida_en"] < fila["order_purchase_timestamp"]) & mismo
    errores += int(conocidos.sum() != fila["seller_n_prev"])
assert errores == 0
```
- Toma 300 pedidos al azar (`sample`).
- Para cada uno (`iterrows` recorre fila por fila) cuenta "a mano" cuántos pedidos **del mismo vendedor** tenían su resultado conocido **antes** de esta compra.
- Compara ese conteo con la columna `seller_n_prev` que calculó la función rápida. Si alguno no coincide, suma un error.
- `assert errores == 0`: si hubiera **un solo** error, el notebook se detiene. Dio 0, así que la variable no hace trampa.

**Para el jurado:** "Comprobamos por fuerza bruta que el historial del vendedor sólo usa información que ya existía el día de la compra."

### Celda 7 (texto) · Carga reciente

### Celda 8 (código) · Gráfica de carga reciente vs. retraso real
```python
semanal = df.set_index("order_purchase_timestamp")[["global_late_30d", "delivery_late"]].resample("W").mean() * 100
ax.plot(... delivery_late ...)    # rojo: lo que realmente pasó
ax.plot(... global_late_30d ...)  # azul: lo que el modelo sabe al comprar
```
- Agrupa por semana (`"W"`) y compara dos líneas: el % real de tardíos de las compras de esa semana (rojo) contra `global_late_30d`, lo que la empresa sabía al momento de comprar (azul).
- La azul va un poco "atrasada", porque sólo usa resultados ya conocidos, pero anticipa las oleadas.

### Celda 9 (código) · Faltantes y guardado
- Tabla con el % de valores faltantes por columna (sólo la distancia tiene ~0.5 %, por códigos postales sin coordenadas). Se rellenarán dentro de cada modelo.
- `describe()` de todas las variables.
- `to_parquet` guarda el dataset final: 96,201 pedidos.

---

## Notebook 04 · Preparación del dataset (split temporal, purga y folds)

**Objetivo:** separar los datos en **entrenamiento** y **prueba (test)** respetando el tiempo, crear los **5 folds** de validación y convertir las categorías en números.
**Entra:** `dataset_pedidos.parquet` · **Sale:** `X_train`, `X_test`, metadatos, `folds.json` y el codificador.

**Analogía general:**
- **Train** = los exámenes pasados con los que estudias.
- **Folds** = exámenes de práctica: estudias con lo más viejo y te pones a prueba con lo siguiente.
- **Test** = el examen final, que **sólo se abre una vez**.
- **Purga** = no estudiar con ejercicios cuya respuesta todavía no te habían dado en esa fecha.

### Celda 1 (texto)
Dibujo del esquema (train, folds, purga, test) y las 4 reglas.

### Celda 2 (código) · Preparación
Igual que siempre, más `joblib` (guarda objetos de Python en archivos) y nuestros módulos `splits` (S) y `preprocesamiento` (P).

### Celda 3 (código) · Corte de test
```python
df = pd.read_parquet(config.DATOS_PROCESSED / "dataset_pedidos.parquet")
idx_train, idx_test, purgados = S.corte_test(df, config.FECHA_CORTE_TEST)
df_train = df.iloc[idx_train].reset_index(drop=True)
df_test = df.iloc[idx_test].reset_index(drop=True)
def fila(d): return {"Filas": len(d), "Desde": ..., "Hasta": ..., "% tardíos": ...}
display(pd.DataFrame({"Train": fila(df_train), "Test": fila(df_test)}).T)
```
- `S.corte_test` regresa las **posiciones** (números de fila) de train y test, y cuántos pedidos se purgaron:
  - **Test:** compras desde el 1 de mayo de 2018.
  - **Train:** compras anteriores **y** cuyo resultado ya se conocía antes de esa fecha.
  - **Purgados:** compras anteriores cuyo resultado se supo después del corte (2,564).
- `iloc[posiciones]` toma esas filas.
- `fila(d)` es una mini-función que resume una tabla: filas, fecha inicial y final, y % de tardíos.

**Resultado:** train 68,287 pedidos (7.6 % tardíos); test 25,350 (4.4 % tardíos).

### Celda 4 (código) · Convertir categorías a números (one-hot)
```python
enc = P.ajustar_codificador(df_train)
X_train, X_test = P.construir_X(df_train, enc), P.construir_X(df_test, enc)
y_train, y_test = df_train["delivery_late"]..., df_test["delivery_late"]...
assert list(X_train.columns) == list(X_test.columns)
```
- Los modelos sólo entienden números. **One-hot** convierte una columna de texto como `customer_state = "SP"` en columnas de sí/no: `customer_state_SP = 1`, `customer_state_RJ = 0`…
- `ajustar_codificador` aprende la lista de categorías **sólo con train** y se queda con las 20 más frecuentes por variable. Las raras se agrupan en "infrequent".
- `construir_X` junta las columnas numéricas con las nuevas columnas 0/1. Resultado: **105 columnas**.
- `X` = las variables (lo que el modelo ve) · `y` = el label (lo que queremos predecir). Es la convención en machine learning.
- El `assert` verifica que train y test tengan exactamente las mismas columnas.

### Celda 5 (texto) · Título de folds

### Celda 6 (código) · Crear los 5 folds con purga
```python
folds = S.folds_temporales(df_train["order_purchase_timestamp"], df_train["label_conocida_en"], config.N_FOLDS)
S.verificar_sin_fuga(folds, ...)
display(S.resumen_folds(folds, ...))
```
- `folds_temporales` parte el train en 6 bloques seguidos en el tiempo:
  - Fold 1: entrena con el bloque 1 y valida con el bloque 2.
  - Fold 2: entrena con los bloques 1–2 y valida con el 3.
  - … y así hasta el fold 5, que entrena con los bloques 1–5 y valida con el 6.
  - Además **purga** del entrenamiento los pedidos cuyo resultado no se conocía al empezar la validación.
- `verificar_sin_fuga` es otra prueba automática: ningún dato de entrenamiento puede ser "del futuro" respecto a su validación.
- `resumen_folds` muestra una tabla con fechas, filas y % de tardíos de cada fold.

### Celda 7 (código) · Gráfica de la línea de tiempo
Dibuja con barras horizontales lo que acabamos de hacer: azul = entrenamiento, naranja = validación, rojo = test. La mini-función `tramo` dibuja un segmento entre dos fechas (`mdates.date2num` convierte fechas a números para poder dibujarlas). Es el **mismo dibujo del pizarrón del profesor**.

### Celda 8 (texto) · Correlación

### Celda 9 (código) · Correlación con el label (sólo train)
```python
corr = X_train.assign(delivery_late=y_train).corr(numeric_only=True)["delivery_late"].drop("delivery_late").dropna()
top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15).sort_values()
```
- La **correlación de Pearson** mide, de −1 a 1, qué tanto se mueven juntas dos columnas. Positiva = cuando una sube, la otra también.
- Se calcula de cada variable contra `delivery_late` y se muestran las 15 más fuertes (en valor absoluto, `abs()`).
- Rojo = aumenta el riesgo de retraso; azul = lo disminuye. Es sólo una primera mirada: la selección real se hace por modelo en el notebook 05.

### Celda 10 (código) · Guardar todo
- `X_train` y `X_test` en parquet.
- Metadatos (ID, fechas y label) de train y test.
- `folds.json`: las posiciones de cada fold, convertidas a lista con `.tolist()` para poder guardarlas en JSON.
- El codificador con `joblib.dump`.
- `split.json`: resumen para la presentación.

---

## Notebooks 05a, 05b, 05c · Modelos clásicos

Los tres notebooks tienen **exactamente la misma estructura** (los generó la misma plantilla). Sólo cambian el modelo, cuántas combinaciones de hiperparámetros se prueban y la gráfica final de interpretación.

| Notebook | Modelo | Combinaciones probadas (`N_ITER`) | Interpretación final |
|---|---|---|---|
| 05a | Regresión Logística | 15 | Coeficientes |
| 05b | Random Forest | 10 | Importancia de variables |
| 05c | XGBoost | 20 | Importancia de variables |

**Los tres modelos en palabras simples:**
- **Regresión Logística:** le pone un "peso" a cada variable, las suma y convierte el resultado en una probabilidad entre 0 y 1 con una curva en forma de S. Es simple y explicable: el signo del peso dice si la variable sube o baja el riesgo.
- **Random Forest (bosque aleatorio):** cientos de árboles de decisión ("¿la distancia es mayor a 1,000 km? → sí → ¿el vendedor tiene historial malo? → …"). Cada árbol ve una parte distinta de los datos y al final **votan**.
- **XGBoost:** también usa árboles, pero en fila: cada árbol nuevo intenta **corregir los errores** de los anteriores (*boosting*).

**¿Qué son los hiperparámetros?** Las "perillas" del modelo que se ajustan **antes** de entrenar: cuántos árboles, qué tan profundos, qué tanta regularización… No se aprenden solas; hay que probar.

### Celda 1 (texto)
Explica el modelo, los 4 pasos (selección de variables → hiperparámetros → evaluación → re-entrenamiento final) y por qué se usa **PR-AUC** y no accuracy: con ~7 % de tardíos, un modelo que siempre diga "a tiempo" tendría ~93 % de accuracy sin detectar ni un retraso.

### Celda 2 (código) · Preparación
```python
from src import modelos as M, evaluacion as E
NOMBRE = "Regresión Logística"   # o "Random Forest" / "XGBoost"
N_ITER = 15
COLOR = viz.COLOR_MODELO[NOMBRE]
```
Trae nuestras funciones de modelos (`M`) y de evaluación (`E`), y define el nombre del modelo, cuántas combinaciones probar y su color en las gráficas (azul, naranja o verde).

### Celda 3 (código) · Cargar datos y folds
```python
X_train = pd.read_parquet(config.DATOS_PROCESSED / "X_train.parquet")
meta_train = pd.read_parquet(config.DATOS_PROCESSED / "meta_train.parquet")
y_train = meta_train["delivery_late"]
folds = [(np.array(f["train"]), np.array(f["val"])) for f in json.loads((config.DATOS_PROCESSED / "folds.json").read_text())]
```
Lee lo que preparó el notebook 04. Los folds venían guardados como listas en JSON; aquí se convierten a arreglos de numpy (`np.array`), que es lo que piden las funciones de scikit-learn. **El test ni siquiera se carga en este notebook.**

### Celda 4 (texto) · Paso 1

### Celda 5 (código) · Paso 1: selección de variables con RFECV
```python
variables, curva_rfecv = M.seleccion_rfecv(NOMBRE, X_train, y_train, folds)
pr_todas = float(curva_rfecv.loc[curva_rfecv["n_variables"].idxmax(), "pr_auc_cv"])
pr_rfecv = float(curva_rfecv["pr_auc_cv"].max())
ax.plot(...); ax.fill_between(...); ax.axvline(len(variables), ...)
```
- **RFECV** = *Recursive Feature Elimination with Cross-Validation* (eliminación recursiva de variables con validación cruzada). Funciona así:
  1. Entrena el modelo con las 105 variables y mide su PR-AUC en los 5 folds.
  2. Quita el 10 % de variables menos importantes **para ese modelo**: la logística usa el tamaño de sus coeficientes; los árboles, su importancia.
  3. Repite hasta llegar a 5 variables.
  4. Se queda con el número de variables que dio el **mejor PR-AUC promedio**.
- Regresa la lista de variables elegidas y la "curva": PR-AUC según el número de variables.
- `pr_todas` = PR-AUC usando todas las variables · `pr_rfecv` = el mejor PR-AUC (con las elegidas).
- La gráfica muestra esa curva, con una banda sombreada (± desviación entre folds) y una línea punteada en el número elegido.

**Analogía:** cocinar un pastel quitando ingredientes de uno en uno y probando si sigue sabiendo bien. Te quedas con la receta más rica.

**Resultado:** Logística 65 variables, Random Forest las 105 (no mejoraba quitando) y XGBoost 35.

### Celda 6 (texto) · Paso 2

### Celda 7 (código) · Paso 2: búsqueda de hiperparámetros
```python
mejores_params, busqueda = M.busqueda_hiperparametros(NOMBRE, X_train[variables], y_train, folds, n_iter=N_ITER)
display(busqueda.head(8).round(4))
```
- Usa `RandomizedSearchCV`: escoge al azar `N_ITER` combinaciones de perillas de una lista de opciones (el "espacio de búsqueda", definido en `src/modelos.py`), evalúa cada una con los 5 folds y se queda con la de mejor PR-AUC.
- Ejemplos de perillas:
  - Logística: `C` (qué tan "libre" es el modelo; un C chico = más regularización = más simple) y `penalty` (tipo de regularización L1 o L2).
  - Random Forest: número de árboles, profundidad máxima, mínimo de pedidos por hoja, cuántas variables mira cada árbol.
  - XGBoost: número de árboles, `learning_rate` (qué tan grandes son los pasos de corrección), profundidad, `subsample`, `scale_pos_weight` (cuánto pesan los tardíos).
- `X_train[variables]` = sólo las columnas que eligió el paso 1.
- La tabla muestra las 8 mejores combinaciones con su score en train y en validación.

**Analogía:** afinar una guitarra probando posiciones de las clavijas y quedándote con la que suena mejor.

### Celda 8 (texto) · Paso 3
Explica que se mide en train **y** en validación, que el sobreajuste se lee con **ROC-AUC** (porque el PR-AUC depende del % de tardíos, que cambia entre periodos) y que el umbral se elige con el F1 máximo.

### Celda 9 (código) · Paso 3: evaluación con los 5 folds
```python
pipe = M.pipeline_con_parametros(NOMBRE, mejores_params)
tabla_cv, oof, umbral = E.evaluar_cv(E.ajustar_predecir_sklearn(pipe), X_train[variables], y_train, folds)
display(tabla_cv.round(4)); display(E.resumen_cv(tabla_cv).round(4))
# gráfica ROC-AUC train vs validación por fold
evolucion = pd.DataFrame({"PR-AUC CV": [pr_todas, pr_rfecv, ..., tabla_cv["PR-AUC val"].mean()]}, index=[...])
```
- `pipeline_con_parametros` arma el modelo final con sus mejores perillas. Un **Pipeline** es una "línea de producción": rellenar faltantes → escalar (sólo la logística) → modelo. Todo se ajusta **dentro** de cada fold, así nunca se usa información de la validación.
- `evaluar_cv` entrena en cada fold y regresa:
  - `tabla_cv`: por fold, PR-AUC y ROC-AUC en train y en validación, % de tardíos, y precisión/recall/F1 de validación.
  - `oof` (*out-of-fold*, "fuera del fold"): la probabilidad que el modelo dio a cada pedido de validación, calculada por un modelo que **no lo vio** al entrenar.
  - `umbral`: el punto de corte que maximiza el F1 sobre esas predicciones.
- `resumen_cv` calcula el promedio y la desviación de cada métrica entre los 5 folds.
- La gráfica une con una línea gris el ROC-AUC de train (gris) y el de validación (color) en cada fold. **Si la distancia es grande, el modelo está memorizando** (sobreajuste).
- `evolucion` muestra cómo mejoró el PR-AUC en cada paso: todas las variables → RFECV → búsqueda → evaluación final.

**¿Qué es el umbral?** El modelo da una probabilidad (por ejemplo 0.63). El umbral es la línea a partir de la cual decimos "alerta: llegará tarde".

### Celda 10 (texto) · Paso 4

### Celda 11 (código) · Paso 4: entrenar con todo el train y guardar
```python
modelo_final = M.pipeline_con_parametros(NOMBRE, mejores_params).fit(X_train[variables], y_train)
carpeta = M.guardar_artefactos(NOMBRE, variables, mejores_params, tabla_cv, oof, umbral, modelo_final, extra={...})
curva_rfecv.to_csv(carpeta / "rfecv_curva.csv", index=False)
busqueda.to_csv(carpeta / "busqueda.csv", index=False)
```
- `.fit(X, y)` = **entrenar**: el modelo aprende de todos los pedidos de entrenamiento.
- `guardar_artefactos` guarda en `modelos/<modelo>/`: las variables elegidas, los hiperparámetros, la tabla del CV, las predicciones fuera de fold, el umbral, un resumen y el modelo entrenado (`modelo.joblib`).
- El notebook 06 usa estos archivos para comparar y hacer la prueba final.

### Celda 12 (texto) · Interpretación

### Celda 13 (código) · Interpretación
- **Logística (05a):** muestra los 15 coeficientes más grandes. Rojo = la variable **sube** el riesgo; azul = lo **baja**. Como las variables están estandarizadas (misma escala), los coeficientes se pueden comparar.
  ```python
  coef = pd.Series(modelo_final.named_steps["modelo"].coef_[0], index=variables)
  ```
  `named_steps["modelo"]` saca el modelo de adentro del Pipeline y `.coef_` son sus pesos.
- **Random Forest y XGBoost (05b, 05c):** muestran las 15 variables más importantes según `feature_importances_` (cuánto ayudó cada variable a separar tardíos de puntuales) y las guardan en `importancias.csv`.

---

## Notebook 05d · Red neuronal en Keras

**Una red neuronal en palabras simples:** capas de "neuronas" conectadas. Cada neurona hace una suma con pesos y luego aplica una regla sencilla (ReLU: "si es negativo, pon 0"). La última capa usa una **sigmoide** que convierte el resultado en una probabilidad entre 0 y 1. La red aprende ajustando los pesos poco a poco para equivocarse menos.

Usamos un **MLP** (perceptrón multicapa) porque nuestros datos son una **tabla**. Las redes convolucionales son para imágenes y las recurrentes para secuencias.

### Celda 1 (texto)
Explica por qué un MLP, cómo se adaptan los pasos del diagrama a una red y cómo se controla el sobreajuste: early stopping con la cola más reciente del train, Dropout y regularización L2.

### Celda 2 (código) · Preparación
Igual que siempre, más `red_neuronal as R`.

### Celda 3 (código) · Cargar datos
Igual que en 05a–c.

### Celda 4 (texto) · Paso 1

### Celda 5 (código) · Paso 1: selección de variables por "conjuntos candidatos"
```python
candidatos = {"Todas": list(X_train.columns)}
for n in config.MODELOS_CLASICOS:
    candidatos[f"RFECV {n}"] = json.loads((config.MODELOS / ... / "best_features.json").read_text())
candidatos["Unión RFECV"] = ...;  candidatos["Intersección RFECV"] = ...
for nombre_set, cols in candidatos.items():
    tabla, oof, umbral = E.evaluar_cv(R.ajustar_predecir_red(cols, R.CONFIG_BASE), X_train, y_train, folds)
...
mejor_set = empatados.sort_values("Variables").iloc[0]["Conjunto"]
```
- RFECV no funciona con redes neuronales, porque una red no tiene "coeficientes" ni "importancias" directas.
- Entonces probamos **6 listas de variables candidatas**: todas, las que eligió cada modelo clásico, la **unión** (todas las que eligió al menos uno) y la **intersección** (las que eligieron los tres).
- Cada lista se evalúa con la red base y los mismos 5 folds.
- Gana la de mejor PR-AUC; si hay empate, la que tiene **menos variables**.

**Resultado:** ganó la lista de XGBoost (35 variables).

### Celda 6 (texto) · Paso 2

### Celda 7 (código) · Paso 2: búsqueda de hiperparámetros de la red
```python
rng = np.random.default_rng(config.RANDOM_STATE)
espacio = {"capas": [(16,), (32,), (32, 16), (64, 32), (64, 32, 16)], "dropout": [...], "l2": [...], "lr": [...], "batch": [...]}
configs = [dict(R.CONFIG_BASE)]
while len(configs) < 6:
    c = {k: v[rng.integers(len(v))] for k, v in espacio.items()}
    if c not in configs: configs.append(c)
```
- `rng` es un generador de números aleatorios con semilla fija (siempre sortea igual).
- El espacio de búsqueda de la red:
  - `capas`: cuántas capas y neuronas, por ejemplo `(64, 32, 16)` = 3 capas de 64, 32 y 16 neuronas.
  - `dropout`: % de neuronas que se "apagan" al azar al entrenar, para que no memorice.
  - `l2`: castigo a los pesos muy grandes.
  - `lr` (*learning rate*): tamaño de cada paso de aprendizaje.
  - `batch`: cuántos pedidos ve antes de actualizar los pesos.
- El `while` sortea configuraciones hasta tener 6 distintas. La primera es la configuración base, que ya se evaluó en el paso 1 y se reutiliza para ahorrar tiempo.
- Cada configuración se evalúa con los 5 folds y gana la de mejor PR-AUC.

**Resultado:** ganó `(64, 32, 16)`, dropout 0.1, L2 0.01, lr 0.0003 y batch 256.

### Celda 8 (texto) · Paso 3

### Celda 9 (código) · Paso 3: tabla del CV y gráfica
Igual que en los modelos clásicos: tabla por fold, promedio ± desviación, umbral y gráfica de ROC-AUC train vs. validación.

### Celda 10 (texto) · Paso 4
Explica que las curvas se miden igual en train y validación. También aclara que si la validación queda **arriba** del train no es un error: la cola reciente (feb–abr 2018) es un periodo de oleadas donde los retrasos son más fáciles de ordenar.

### Celda 11 (código) · Entrenamiento final y curvas de aprendizaje
```python
red_final = R.RedTabular(columnas, mejor_cfg, curvas=True).fit(X_train, y_train)
reg = red_final.cb_curvas.registro
axes[0].plot(epocas, reg["loss_train"]) ...   # pérdida
axes[1].plot(epocas, reg["auc_train"]) ...    # ROC-AUC
red_final.red.summary()
```
- `RedTabular` es nuestra "envoltura" de la red (ver [Anexo A](#red_neuronalpy)). Al hacer `.fit`:
  1. Separa el 15 % más reciente del train para el early stopping.
  2. Rellena faltantes y escala las variables.
  3. Entrena época por época. Una **época** = una pasada completa por todos los datos.
  4. **Early stopping:** si durante 8 épocas la pérdida de validación no mejora, se detiene y regresa a la mejor época (aquí la 58).
- `curvas=True` registra, al final de cada época, la pérdida y el ROC-AUC en train y validación. Con eso se dibujan las **curvas de aprendizaje**.
- `summary()` imprime la arquitectura: capas, neuronas y número de parámetros.

### Celda 12 (código) · Guardar
Guarda lo mismo que los modelos clásicos y además la red en `red.keras` y el preprocesamiento en `prep.joblib` (con `R.guardar_red`).

---

## Notebook 06 · Selección del mejor modelo y prueba final

**Objetivo:** comparar los 4 modelos **sólo con el CV**, elegir y luego abrir el test **una sola vez**.

### Celda 1 (texto)
Los 4 pasos: comparar en CV → elegir el mejor clásico (empate → el más simple) → prueba final (mejor clásico + red) → traducción a negocio.

### Celda 2 (código) · Preparación
Igual que siempre, más:
- `precision_recall_curve`: puntos de la curva precisión-recall.
- `ConfusionMatrixDisplay`: dibuja la matriz de confusión.
- Nuestros módulos `evaluacion` (E) y `red_neuronal` (R).

### Celda 3 (código) · Cargar resultados y tabla comparativa
```python
TODOS = config.MODELOS_CLASICOS + [config.NOMBRE_RED]
resumen = {n: json.loads((carpeta[n] / "resumen.json").read_text()) for n in TODOS}
cv = {n: pd.read_csv(carpeta[n] / "cv_folds.csv") for n in TODOS}
oof = {n: pd.read_parquet(carpeta[n] / "oof.parquet") for n in TODOS}
comp = pd.DataFrame({n: {"Variables": ..., "PR-AUC val (prom)": ..., "ROC-AUC train (prom)": ..., "Brecha ROC train-val": ..., ...} for n in TODOS}).T.astype(float)
comp["PR-AUC val / azar"] = comp["PR-AUC val (prom)"] / azar_cv
```
- Lee lo que guardó cada notebook 05.
- Arma la tabla comparativa: número de variables, PR-AUC de validación (promedio y desviación), ROC-AUC de validación y de train, **brecha** de ROC (train − validación = sobreajuste), F1 fuera de fold y umbral.
- `azar_cv` = % de tardíos en validación (8.3 %). Un modelo que adivina al azar obtiene un PR-AUC igual a ese porcentaje. Por eso `PR-AUC / azar` dice **cuántas veces mejor que el azar** es cada modelo.

### Celda 4 (código) · Figura comparativa (3 paneles)
1. **Izquierda:** PR-AUC de validación con barras de error (± desviación). Es la métrica para **elegir**.
2. **Centro:** ROC-AUC de train (gris) vs. validación (color). Es la gráfica para ver el **sobreajuste**: Random Forest y XGBoost tienen la brecha más grande.
3. **Derecha:** curvas precisión-recall con las predicciones de los 5 folds juntas.

### Celda 5 (texto) · Cómo leer la curva de la derecha
Cada fold es un modelo distinto con su propia escala de probabilidades. Por eso, juntarlos puede hacer que un modelo se vea peor de lo que es (le pasa a la red). La selección usa el **promedio por fold**.

### Celda 6 (texto) · Selección

### Celda 7 (código) · Elegir modelos (sólo con el CV)
```python
pr_cv = {n: comp.loc[n, "PR-AUC val (prom)"] for n in TODOS}
mejor_clasico = E.mejor_modelo({n: pr_cv[n] for n in config.MODELOS_CLASICOS}, config.MODELOS_CLASICOS)
ganador = E.mejor_modelo(pr_cv, config.MODELOS_CLASICOS + [config.NOMBRE_RED])
A_TEST = [mejor_clasico, config.NOMBRE_RED]
```
- `mejor_modelo` aplica la regla fijada **antes** de ver resultados: gana el de mayor PR-AUC, pero si otro más simple queda a menos de 0.005, gana el más simple.
- Los tres clásicos quedaron a menos de 0.005 entre sí → gana la **Logística** (la más simple).
- Entre los 4, la red gana el CV (0.190).
- `A_TEST` = los dos modelos que van a la prueba final: el mejor clásico y la red.

### Celda 8 (texto) · Prueba final

### Celda 9 (código) · Abrir el test (una sola vez)
```python
X_test = pd.read_parquet(...); y_test = meta_test["delivery_late"]
modelos[n] = R.RedCargada(carpeta[n])  # la red
modelos[n] = joblib.load(carpeta[n] / "modelo.joblib")  # los clásicos
prob_test = {n: modelos[n].predict_proba(X_test[variables[n]])[:, 1] ...}
res_test = pd.DataFrame({n: {**E.metricas(y_test, prob_test[n], resumen[n]["umbral"]), ...} for n in A_TEST}).T.astype(float)
res_test["PR-AUC / azar"] = res_test["PR-AUC"] / azar_test
```
- Ésta es la **primera y única vez** que se cargan los datos de test.
- Carga los modelos ya entrenados (no se vuelven a entrenar).
- `predict_proba` da, para cada pedido, la probabilidad de cada clase. `[:, 1]` toma la columna de la clase 1 (tardío).
- `E.metricas` calcula PR-AUC, ROC-AUC, precisión, recall, F1 y accuracy usando el umbral que cada modelo eligió en validación.
- `**diccionario` "desempaca" un diccionario dentro de otro.

**Resultado:** Logística PR-AUC 0.098 y red 0.095; ambos ≈ 2.2 veces el azar (4.4 %).

### Celda 10 (código) · Matrices de confusión y curvas PR en test
- **Matriz de confusión:** tabla de 2×2 con aciertos y errores:
  - Verdaderos negativos (dijo "a tiempo" y fue a tiempo).
  - Falsos positivos (dio alerta y llegó a tiempo).
  - Falsos negativos (no dio alerta y llegó tarde).
  - Verdaderos positivos (dio alerta y llegó tarde).
- **Curva precisión-recall:** cómo cambia la precisión conforme detectamos más tardíos. La línea punteada es el azar.

### Celda 11 (texto) · Nota sobre el umbral
El umbral elegido en validación marca demasiados pedidos en test (la tasa de retraso bajó). Por eso conviene operar "por capacidad": revisar el *k* % más riesgoso.

### Celda 12 (texto) · Traducción a negocio

### Celda 13 (código) · Captura de tardíos y curva de captura
```python
captura = {n: E.captura_top_k(y_test, prob_test[n]) for n in A_TEST}
y_ord = y_test.to_numpy()[np.argsort(-prob_test[n])]
curva = np.concatenate([[0], np.cumsum(y_ord)]) / y_ord.sum()
```
- `captura_top_k` responde: "si reviso el 5 %, 10 %, 20 % o 30 % de pedidos con más riesgo, ¿qué % de los tardíos atrapo?". También da la precisión y el **lift** (cuántas veces mejor que revisar al azar).
- `np.argsort(-prob)` ordena los pedidos de mayor a menor riesgo. `np.cumsum` va sumando cuántos tardíos llevamos. Dividir entre el total da la **curva de captura**.

**Resultado:** revisando el 20 % de pedidos se anticipa el 39 % de los retrasos (Logística).

### Celda 14 (texto) · Variables importantes

### Celda 15 (código) · Importancia del mejor modelo clásico
Si el modelo tiene `feature_importances_` (árboles), las usa. Si no (logística), usa el valor absoluto de los coeficientes (`np.abs(coef_)`). Las normaliza para que sumen 100 % y grafica las 12 principales: días prometidos, retrasos del sistema en 30 días, pedidos de los últimos 7 días…

### Celda 16 (texto) · Lectura de resultados y limitaciones

### Celda 17 (código) · Exportar resultados para la presentación
- `curva_pr` y `curva_captura` son mini-funciones que reducen las curvas a pocos puntos (para que el archivo pese poco).
- Arma el diccionario `resultados` con todo (tabla del CV, resultados del test, captura, curvas, importancias, matrices de confusión) y lo guarda en `reportes/metricas/resultados.json`.
- La presentación y el artículo leen este archivo, así que **ningún número está escrito a mano**.

### Celda 18 (texto) · Conclusión

### Celda 19 (código) · Conclusión automática
Arma con código un texto con las conclusiones, usando los números reales:
- Ganador en CV.
- Resultados en test.
- Si la diferencia en test es menor que 0.01, dice que ambos modelos son equivalentes y recomienda la logística por simple y explicable, con la red como "retadora".

---

## Anexo A · Las funciones de `src/` por dentro

### `limpieza.py`

| Función | Qué hace |
|---|---|
| `cargar_tablas()` | Lee los 8 CSV y regresa un diccionario `{nombre: tabla}`. |
| `_zip5(s)` | Convierte un código postal a texto de 5 dígitos (`zfill(5)`). El guion bajo inicial indica que es una función "de uso interno". |
| `limpiar_customers(df)` / `limpiar_sellers(df)` | Código postal a 5 dígitos y filtro de estados oficiales de Brasil. |
| `limpiar_reviews(df)` | Fechas a formato fecha; borra título y comentario. |
| `limpiar_items(df)` | `shipping_limit_date` a fecha. |
| `limpiar_products(df)` | Borra 3 columnas, peso 0 → faltante, categoría vacía → `"sin_categoria"`. |
| `limpiar_geolocation(df)` | Código postal a 5 dígitos y coordenadas dentro de Brasil. |
| `limpiar_orders(df)` | Fechas; quita cancelados, sin entrega y censurados; regresa la tabla y la bitácora. |
| `definir_objetivo(orders)` | Crea `delivery_late`, `label_conocida_en` y `dias_retraso_entrega`. `np.minimum(a, b)` toma, fila por fila, la menor de dos fechas. |

### `features.py`

**`haversine_km(lat1, lon1, lat2, lon2)`**: distancia en km entre dos puntos de la Tierra.
- Convierte los grados a radianes (`np.radians`) y aplica la fórmula de Haversine con senos y cosenos. 6371 es el radio de la Tierra en km.
- Se usa porque la Tierra es redonda: una regla recta sobre un mapa plano se equivocaría.

**`_eventos(hist, clave)`**: prepara la "lista de resultados conocidos".
- Ordena los pedidos por el momento en que se supo su resultado (`label_conocida_en`).
- Para cada clave (por ejemplo, cada vendedor) calcula, en orden: cuántos resultados se conocían hasta ese momento (`cumcount() + 1`) y cuántos eran tardíos (`cumsum` del label). Es un **contador acumulado**.

**`_acumulado_antes(consultas, eventos, clave, desfase, estricto)`**: el corazón del "point-in-time".
- Usa `pd.merge_asof`: para cada compra, busca **la última fila del contador acumulado anterior a la fecha de compra** del mismo vendedor (o ruta).
- `direction="backward"` = busca hacia atrás en el tiempo.
- `allow_exact_matches=False` = estrictamente antes (no el mismo instante).
- Si no hay nada antes, pone 0 (`fillna(0)`).
- **Analogía:** "¿cuántas veces había fallado este vendedor, según lo que sabíamos justo antes de que el cliente comprara?".

**`tasa_global_previa(hist, consultas)`**: la tasa de retraso de **todo** Olist conocida antes de cada compra. Sirve como valor de referencia (*prior*) para el suavizado. Usa el mismo truco con una clave falsa `"_todo"` para que todos estén en el mismo grupo.

**`tasa_historica(hist, consultas, clave, nombre, prior, m=20)`**: tasa de retraso suavizada de un vendedor o ruta.
- Fórmula: `(tardíos + m × prior) / (pedidos + m)`.
- **¿Por qué suavizar?** Un vendedor con 1 pedido tardío de 1 tendría 100 % de retraso, lo cual exagera. El suavizado "mezcla" su tasa con la tasa general como si tuviera 20 pedidos imaginarios promedio. Con muchos pedidos reales, manda su propia historia; con pocos, se parece al promedio.
- Regresa dos columnas: `..._late_rate` (la tasa) y `..._n_prev` (cuántos pedidos previos conocidos).

**`tasa_reciente(hist, consultas, clave, nombre, prior, dias=30, m=20)`**: tasa de retraso **de los últimos 30 días**.
- Calcula el acumulado hasta justo antes de la compra (`hasta`) y el acumulado hasta 30 días antes (`desde`).
- La resta da cuántos resultados y cuántos tardíos hubo **en esa ventana de 30 días**. Después suaviza igual.
- Con `clave=None` es para todo Olist (`global_late_30d`); con `"customer_state"`, para el estado del cliente (`estado_late_30d`).

**`construir_dataset(t, orders_raw)`**: arma la tabla final, una fila por pedido. Paso a paso:
1. Columnas del pedido: `purchase_weekday` (0 = lunes … 6 = domingo), `franja_horaria` con `pd.cut` (corta las horas en 4 rangos), `estimated_delivery_days` (fecha prometida − fecha de compra, en días).
2. `merge` con clientes para el estado y código postal (`validate="one_to_one"` comprueba que no se dupliquen filas).
3. **Ítems:** `merge` de ítems + productos + vendedores. `dias_limite_envio` = días entre la compra y la fecha límite de despacho. Después `groupby("order_id").agg(...)` resume cada pedido: cuenta ítems, suma precios y fletes, promedia pesos y medidas, toma el primer vendedor y la primera categoría, etc.
4. **Pagos:** total, máximo de cuotas, tipos distintos y el tipo más frecuente (`mode()` = moda).
5. **Distancia:** promedia las coordenadas por código postal, las pega a clientes y vendedores, y calcula Haversine para cada ítem. Por pedido: distancia promedio y máxima.
6. **Ruta:** `same_state` (1 si vendedor y cliente están en el mismo estado), `ruta` (texto "estado vendedor_estado cliente") y `freight_ratio` = flete / precio (el `1e-6` evita dividir entre 0).
7. Quita pedidos sin ítems o sin pago.
8. Ordena por fecha de compra y calcula las variables históricas y de carga reciente con las funciones anteriores (`.join` las pega por `order_id`).
9. **`pedidos_ult_7d`:** con `np.searchsorted` sobre la lista ordenada de todas las compras, cuenta cuántas compras hubo en los 7 días anteriores. Es como buscar en un diccionario ordenado por fecha: la posición de "hoy" menos la posición de "hace 7 días".
10. `assert`: que cada pedido aparezca una vez y que esté en orden de fecha.

### `splits.py`

| Función | Qué hace |
|---|---|
| `corte_test(df, fecha)` | Máscaras de verdadero/falso: test = compra ≥ fecha; train = compra < fecha **y** label conocido < fecha; purgados = el resto de los anteriores. `np.flatnonzero` convierte la máscara en posiciones. |
| `folds_temporales(compra, conocida, n_folds)` | `np.array_split` parte el train en 6 bloques iguales. Para cada fold k: validación = bloque k+1; entrenamiento = bloques anteriores cuyo label se conocía antes del inicio de la validación. Regresa una lista de pares (posiciones de train, posiciones de validación). |
| `resumen_folds(...)` | Tabla con fechas, filas y % de tardíos de cada fold. |
| `verificar_sin_fuga(...)` | `assert`: el último label conocido de entrenamiento es anterior a la primera compra de validación. |

### `preprocesamiento.py`

| Elemento | Qué hace |
|---|---|
| `COLUMNAS_CATEGORICAS` | Las 6 columnas de texto que se convierten con one-hot. |
| `COLUMNAS_EXCLUIR` | IDs, fechas y el label: **nunca** entran al modelo. Meter el ID del pedido sería absurdo, y meter el label sería hacer trampa. |
| `ajustar_codificador(df_train)` | Crea un `OneHotEncoder` de scikit-learn (máximo 20 categorías por variable; las desconocidas o raras van a "infrequent") y lo ajusta **sólo** con train. |
| `construir_X(df, enc)` | Numéricas + columnas 0/1 del one-hot. `_limpiar_nombre` quita caracteres raros de los nombres de columnas (XGBoost no acepta `[`, `]` ni `<`). |

### `modelos.py`

| Elemento | Qué hace |
|---|---|
| `_imputador()` | `SimpleImputer(strategy="median")`: rellena faltantes con la mediana (aprendida en cada fold). `keep_empty_features=True` evita que borre columnas vacías. |
| `crear_pipeline(nombre, rapido)` | Arma la línea de producción de cada modelo. **Logística:** imputador → `StandardScaler` (pone cada variable con promedio 0 y desviación 1) → `LogisticRegression(class_weight="balanced")`. **Random Forest:** imputador → bosque. **XGBoost:** sólo el modelo (maneja faltantes solo). `rapido=True` usa menos árboles para que el RFECV no tarde tanto. |
| `class_weight="balanced"` / `scale_pos_weight` | Le dicen al modelo que un tardío "pesa más" que un puntual, porque hay muy pocos. Si no, el modelo aprendería a decir siempre "a tiempo". |
| `IMPORTANCIA` | Le dice al RFECV dónde buscar la importancia de cada variable: `coef_` (logística) o `feature_importances_` (árboles). |
| `ESPACIOS` | Las opciones de cada perilla para la búsqueda. `loguniform(1e-3, 1e1)` = sortear valores entre 0.001 y 10 de forma pareja en escala logarítmica. |
| `N_JOBS_CV` | Cuántas tareas en paralelo: la logística usa 1 núcleo por modelo, así que se corren 5 folds a la vez; los árboles ya usan todos los núcleos. |
| `seleccion_rfecv(...)` | Corre `RFECV(step=0.1, cv=folds, scoring="average_precision", min_features_to_select=5)` y regresa las variables y la curva. |
| `busqueda_hiperparametros(...)` | Corre `RandomizedSearchCV(..., refit=False, return_train_score=True)` y regresa los mejores parámetros y la tabla ordenada. |
| `pipeline_con_parametros(nombre, params)` | Crea el pipeline y le pone los parámetros con `set_params`. Los nombres como `modelo__C` significan "el parámetro C del paso llamado modelo". |
| `guardar_artefactos(...)` | Escribe los JSON, CSV, parquet y el `modelo.joblib` en la carpeta del modelo. |

### `evaluacion.py`

| Función | Qué hace |
|---|---|
| `umbral_optimo(y, prob)` | `precision_recall_curve` da la precisión y el recall para cada posible umbral; calcula el F1 = 2·P·R/(P+R) de cada uno y regresa el umbral con el F1 máximo (`np.nanargmax`). |
| `metricas(y, prob, umbral)` | Convierte probabilidades en predicciones 0/1 (`prob >= umbral`) y calcula PR-AUC, ROC-AUC, precisión, recall, F1 y accuracy. |
| `evaluar_cv(ajustar_predecir, X, y, folds)` | Para cada fold: entrena, predice en train y validación y guarda las métricas y las predicciones fuera de fold. Al final elige el umbral y calcula precisión/recall/F1 por fold. |
| `ajustar_predecir_sklearn(pipeline)` | Crea una función que **clona** el pipeline (copia limpia sin entrenar), lo entrena y regresa las probabilidades. Clonar evita que un fold "contamine" al siguiente. |
| `resumen_cv(tabla)` | Promedio y desviación de cada métrica entre folds. |
| `captura_top_k(y, prob)` | Ordena por riesgo y calcula, para el 5/10/20/30 % superior, el % de tardíos capturados, la precisión y el lift. |
| `mejor_modelo(pr_auc_cv, orden_simple, tol)` | Recorre los modelos de más simple a más complejo y regresa el primero que quede a menos de `tol` del mejor. |

### `red_neuronal.py`

| Elemento | Qué hace |
|---|---|
| `CONFIG_BASE` | Configuración inicial: capas (32, 16), dropout 0.3, L2 0.001, lr 0.0005 y batch 512. |
| `pesos_balanceados(y)` | Calcula cuánto debe pesar cada clase para que tardíos y puntuales "valgan lo mismo" en total. |
| `crear_red(...)` | `keras.Sequential`: capas en fila. Por cada capa: `Dense(n, activation="relu", kernel_regularizer=l2)` + `Dropout`. Al final, `Dense(1, activation="sigmoid")`. Se compila con el optimizador **Adam** (ajusta los pesos de forma inteligente) y la pérdida **binary_crossentropy** (mide qué tan lejos están las probabilidades de la respuesta real). |
| `CurvasEnInferencia` | Un *callback* (código que Keras ejecuta al final de cada época). Mide la pérdida y el ROC-AUC en train y validación **sin Dropout**, para que las curvas se puedan comparar. |
| `RedTabular` | Envoltura con `fit` y `predict_proba`, igual que scikit-learn, para usar la red con las mismas funciones de evaluación. `fit`: fija la semilla, separa el 15 % más reciente para early stopping, ajusta imputación y escalado, crea la red, entrena con `EarlyStopping(patience=8, restore_best_weights=True)` y guarda la mejor época. |
| `ajustar_predecir_red(columnas, cfg)` | Igual que `ajustar_predecir_sklearn`, pero para la red. |
| `guardar_red` / `RedCargada` | Guardan y cargan la red (`red.keras`) y su preprocesamiento (`prep.joblib`). |

### `viz.py`
- `COLOR_MODELO`: un color fijo por modelo (Logística azul, Random Forest naranja, XGBoost verde, Red amarillo). Rojo = tardío.
- `aplicar_estilo()`: configura matplotlib (fondos, letras, sin bordes arriba ni a la derecha, cuadrícula suave).
- `guardar(fig, nombre)`: guarda la figura como PNG en `reportes/figuras/`.

---

## Anexo B · Glosario de funciones de librerías

### pandas (`pd`) · tablas
| Función | Qué hace | Ejemplo mental |
|---|---|---|
| `pd.read_csv` / `pd.read_parquet` | Leer una tabla de un archivo | Abrir un Excel |
| `df.to_parquet` / `df.to_csv` | Guardar una tabla | "Guardar como" |
| `df["col"]` | Tomar una columna | Seleccionar la columna B |
| `df[condición]` | Filtrar filas | Filtro de Excel |
| `df.loc[filas, cols]` / `df.iloc[posiciones]` | Seleccionar por nombre / por posición | Celdas específicas |
| `df.merge(otra, on="id", how="left")` | Unir dos tablas por una columna común | BUSCARV |
| `df.join(otra, on="id")` | Igual que merge, usando el índice de la otra tabla | BUSCARV |
| `df.groupby("col").agg(...)` | Agrupar y resumir | Tabla dinámica |
| `value_counts(normalize=True)` | Contar valores (en proporción) | Conteo de frecuencias |
| `describe()` | Resumen estadístico | Promedio, mín, máx… |
| `isna()` / `dropna()` / `fillna()` | Detectar / quitar / rellenar faltantes | Celdas vacías |
| `pd.to_datetime` | Convertir texto a fecha | Formato de fecha |
| `.dt.normalize()` / `.dt.dayofweek` / `.dt.hour` | Poner la hora en 00:00 / día de la semana / hora | Funciones de fecha |
| `resample("MS")` | Agrupar por periodos de tiempo | Total por mes |
| `pd.merge_asof` | Unir con "el último registro anterior a esta fecha" | Buscar el valor vigente en esa fecha |
| `pd.cut` | Cortar números en rangos | Hora → franja horaria |
| `sort_values` / `reset_index` | Ordenar / renumerar filas | Ordenar A→Z |
| `assign` | Agregar columnas nuevas | Nueva columna con fórmula |
| `corr()` | Correlación entre columnas | ¿Se mueven juntas? |
| `cumcount` / `cumsum` | Contador / suma acumulados | Total acumulado |

### numpy (`np`) · números
| Función | Qué hace |
|---|---|
| `np.array` | Lista de números rápida |
| `np.mean`, `np.median` | Promedio y mediana |
| `np.where(cond, a, b)` | "Si cond entonces a, si no b", elemento por elemento |
| `np.argsort` | Posiciones que ordenarían los datos |
| `np.cumsum` | Suma acumulada |
| `np.searchsorted` | Dónde entraría un valor en una lista ordenada |
| `np.array_split` | Partir en pedazos casi iguales |
| `np.radians`, `np.sin`, `np.cos`, `np.arcsin` | Trigonometría (para Haversine) |
| `np.random.default_rng(semilla)` | Generador de azar reproducible |

### matplotlib (`plt`) · gráficas
| Función | Qué hace |
|---|---|
| `plt.subplots` | Crear el lienzo y los ejes |
| `ax.plot` / `ax.bar` / `ax.barh` / `ax.hist` / `ax.scatter` / `ax.boxplot` | Línea / barras / barras horizontales / histograma / puntos / caja |
| `ax.axvline` / `ax.axhline` | Línea vertical / horizontal de referencia |
| `ax.fill_between` | Sombrear una banda (± desviación) |
| `ax.set_title`, `set_xlabel`, `set_ylabel`, `legend` | Título, nombres de ejes y leyenda |
| `fig.savefig` | Guardar la imagen |

### scikit-learn · machine learning
| Herramienta | Qué hace |
|---|---|
| `Pipeline` | Encadena pasos (rellenar → escalar → modelo) que se ajustan juntos |
| `SimpleImputer` | Rellena faltantes (mediana) |
| `StandardScaler` | Pone cada variable con promedio 0 y desviación 1 |
| `OneHotEncoder` | Categorías → columnas 0/1 |
| `LogisticRegression` | Regresión logística |
| `RandomForestClassifier` | Bosque aleatorio |
| `RFECV` | Selección de variables quitando las menos importantes |
| `RandomizedSearchCV` | Prueba combinaciones de hiperparámetros al azar |
| `clone` | Copia un modelo sin entrenar |
| `average_precision_score` | PR-AUC |
| `roc_auc_score` | ROC-AUC |
| `precision_score`, `recall_score`, `f1_score`, `accuracy_score` | Métricas con umbral |
| `precision_recall_curve` | Puntos de la curva precisión-recall |
| `ConfusionMatrixDisplay` | Dibuja la matriz de confusión |
| `.fit(X, y)` / `.predict_proba(X)` | Entrenar / dar probabilidades |

### XGBoost y Keras
| Herramienta | Qué hace |
|---|---|
| `XGBClassifier` | Modelo de boosting de árboles |
| `keras.Sequential` | Red neuronal de capas en fila |
| `layers.Dense` | Capa de neuronas totalmente conectada |
| `layers.Dropout` | Apaga neuronas al azar al entrenar |
| `regularizers.l2` | Castigo a pesos grandes |
| `keras.optimizers.Adam` | Algoritmo que ajusta los pesos |
| `keras.callbacks.EarlyStopping` | Detiene el entrenamiento cuando ya no mejora |
| `model.fit(..., epochs, batch_size, class_weight)` | Entrenar la red |
| `model.save` / `keras.models.load_model` | Guardar / cargar la red |

### Métricas en una línea
| Métrica | Pregunta que responde |
|---|---|
| **Precisión** | De los pedidos que marqué como tardíos, ¿cuántos de verdad lo fueron? |
| **Recall** | De todos los tardíos reales, ¿cuántos atrapé? |
| **F1** | Equilibrio entre precisión y recall (promedio "armónico"). |
| **Accuracy** | ¿Qué % de todas las predicciones fue correcto? (engaña con clases desbalanceadas). |
| **ROC-AUC** | Si tomo un tardío y un puntual al azar, ¿qué tan seguido el tardío tiene más riesgo? 0.5 = azar, 1 = perfecto. |
| **PR-AUC** | ¿Qué tan bien pongo a los tardíos hasta arriba de la lista? El azar = % de tardíos. |
| **Lift / "× azar"** | ¿Cuántas veces mejor que elegir al azar? |
