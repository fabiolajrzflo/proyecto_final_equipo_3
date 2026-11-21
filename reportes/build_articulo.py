"""Genera el reporte técnico (artículo científico corto) en HTML y PDF con los resultados reales.

Uso (desde la raíz del proyecto):  .venv/bin/python reportes/build_articulo.py
Requiere haber corrido los notebooks 01–06.
"""
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import pandas as pd  # noqa: E402
from src import config  # noqa: E402
from src.preprocesamiento import COLUMNAS_EXCLUIR  # noqa: E402

M = config.METRICAS
eda = json.loads((M / "eda.json").read_text())
split = json.loads((M / "split.json").read_text())
limp = json.loads((M / "limpieza.json").read_text())
res = json.loads((M / "resultados.json").read_text())
params = {n: json.loads((config.MODELOS / c / "best_params.json").read_text()) for n, c in config.CLAVES_MODELO.items()}
resumen = res["resumen_modelos"]
cols = pd.read_parquet(config.DATOS_PROCESSED / "dataset_pedidos.parquet").columns
n_base = sum(c not in COLUMNAS_EXCLUIR for c in cols)

NN = config.NOMBRE_RED
mejor = res["mejor_clasico"]
ganador = res["ganador_cv"]
cv = {r["Modelo"]: r for r in res["comparacion_cv"]}
te = {r["Modelo"]: r for r in res["test"]}
cap = {int(round(r["% pedidos revisados"])): r for r in res["captura"][mejor]}
cap_nn = {int(round(r["% pedidos revisados"])): r for r in res["captura"][NN]}


def f(x, d=3):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", " ")


def p(x, d=1):
    return f(x * 100, d) + " %"


def n(x):
    return f"{int(x):,}".replace(",", " ")


orden = sorted(cv, key=lambda m: -cv[m]["PR-AUC val (prom)"])
mejor_pr = max(cv[m]["PR-AUC val (prom)"] for m in config.MODELOS_CLASICOS)
empate = [m for m in config.MODELOS_CLASICOS if cv[m]["PR-AUC val (prom)"] >= mejor_pr - config.TOLERANCIA_EMPATE]
frase_empate = (
    f"Los modelos {', '.join(empate[:-1])} y {empate[-1]} quedaron a menos de {f(config.TOLERANCIA_EMPATE, 3)} de PR-AUC entre sí; "
    f"por la regla de simplicidad definida antes de ver los resultados, el mejor modelo clásico es <b>{mejor}</b>."
    if len(empate) > 1 else f"El mejor modelo clásico es <b>{mejor}</b>.")
nn_vs = cv[NN]["PR-AUC val (prom)"] - cv[mejor]["PR-AUC val (prom)"]
frase_nn = ("La red neuronal superó al mejor modelo clásico en validación" if nn_vs > config.TOLERANCIA_EMPATE else
            "La red neuronal no superó al mejor modelo clásico en validación" if nn_vs < -config.TOLERANCIA_EMPATE else
            "La red neuronal quedó empatada con el mejor modelo clásico en validación")
mejor_test = mejor if te[mejor]["PR-AUC"] >= te[NN]["PR-AUC"] else NN
dif_test = te[NN]["PR-AUC"] - te[mejor]["PR-AUC"]
paridad = abs(dif_test) < 0.01
conf = res["confusion"][mejor]
marcados = (conf["tp"] + conf["fp"]) / sum(conf.values())
frase_test = (
    f"En prueba ambos modelos resultaron prácticamente equivalentes (PR-AUC {f(te[mejor]['PR-AUC'])} y {f(te[NN]['PR-AUC'])}; "
    f"ROC-AUC {f(te[mejor]['ROC-AUC'])} y {f(te[NN]['ROC-AUC'])}): la ventaja de la red en validación no se sostuvo en el futuro."
    if paridad else f"{mejor_test} obtuvo el mayor PR-AUC en prueba.")
frase_conclusion = (
    f"Bajo ese diseño, los tres modelos clásicos quedaron empatados y {mejor} los representa por simplicidad; la red neuronal fue la mejor "
    f"en validación, pero en el periodo de prueba ambos fueron equivalentes. Por explicabilidad y costo recomendamos {mejor} para un piloto, "
    f"con la red como modelo retador." if paridad else
    f"Bajo ese diseño, {mejor_test} fue el modelo con mejor desempeño en el periodo de prueba.")


def tabla_cv():
    filas = "".join(
        f"<tr{' class=win' if m == mejor else ''}><td>{m}</td><td>{int(cv[m]['Variables'])}</td>"
        f"<td>{f(cv[m]['PR-AUC val (prom)'])} ± {f(cv[m]['PR-AUC val (desv)'])}</td><td>{f(cv[m]['PR-AUC val / azar'], 2)}</td>"
        f"<td>{f(cv[m]['ROC-AUC val (prom)'])}</td><td>{f(cv[m]['ROC-AUC train (prom)'])}</td><td>{f(cv[m]['Brecha ROC train-val'])}</td>"
        f"<td>{f(cv[m]['F1 fuera de fold'])}</td></tr>" for m in orden)
    return ("<table><thead><tr><th>Modelo</th><th>Variables</th><th>PR-AUC val</th><th>× azar</th><th>ROC-AUC val</th>"
            "<th>ROC-AUC train</th><th>Brecha ROC</th><th>F1 (OOF)</th></tr></thead><tbody>" + filas + "</tbody></table>")


def tabla_test():
    filas = "".join(
        f"<tr{' class=win' if m == mejor_test else ''}><td>{m}</td><td>{f(te[m]['PR-AUC CV'])}</td><td>{f(te[m]['PR-AUC'])}</td>"
        f"<td>{f(te[m]['PR-AUC / azar'], 2)}</td><td>{f(te[m]['ROC-AUC'])}</td><td>{f(te[m]['Precision'])}</td>"
        f"<td>{f(te[m]['Recall'])}</td><td>{f(te[m]['F1'])}</td><td>{f(te[m]['Umbral'], 2)}</td></tr>" for m in [mejor, NN])
    return ("<table><thead><tr><th>Modelo</th><th>PR-AUC CV</th><th>PR-AUC test</th><th>× azar</th><th>ROC-AUC</th>"
            "<th>Precisión</th><th>Recall</th><th>F1</th><th>Umbral</th></tr></thead><tbody>" + filas + "</tbody></table>")


def tabla_folds():
    filas = "".join(
        f"<tr><td>{r['Fold']}</td><td>{r['Train desde'][:10]} – {r['Train hasta'][:10]}</td><td>{n(r['Filas train'])}</td>"
        f"<td>{r['Val desde'][:10]} – {r['Val hasta'][:10]}</td><td>{n(r['Filas val'])}</td>"
        f"<td>{f(r['% tardíos train'], 2)}</td><td>{f(r['% tardíos val'], 2)}</td></tr>" for r in split["folds"])
    return ("<table><thead><tr><th>Fold</th><th>Periodo de entrenamiento</th><th>Filas</th><th>Periodo de validación</th>"
            "<th>Filas</th><th>% tardíos train</th><th>% tardíos val</th></tr></thead><tbody>" + filas + "</tbody></table>")


def tabla_captura():
    filas = "".join(
        f"<tr><td>{k} %</td><td>{f(cap[k]['% tardíos capturados'], 1)} %</td><td>{f(cap[k]['Precision'] * 100, 1)} %</td>"
        f"<td>{f(cap[k]['Lift'], 2)}</td><td>{f(cap_nn[k]['% tardíos capturados'], 1)} %</td></tr>" for k in sorted(cap))
    return (f"<table><thead><tr><th>Pedidos revisados</th><th>Tardíos capturados ({mejor})</th><th>Precisión</th><th>Lift</th>"
            f"<th>Tardíos capturados (red)</th></tr></thead><tbody>" + filas + "</tbody></table>")


def fmt_param(v):
    if isinstance(v, float):
        return f(v, 4) if abs(v) < 1 else f(v, 2)
    if isinstance(v, list):
        return "(" + ", ".join(map(str, v)) + ")"
    return str(v)


def tabla_params():
    filas = ""
    for m in config.CLAVES_MODELO:
        ps = "; ".join(f"{k.replace('modelo__', '')} = {fmt_param(v)}" for k, v in params[m].items())
        extra = f" (conjunto: {resumen[m].get('conjunto_variables')})" if m == NN else ""
        filas += f"<tr><td>{m}</td><td>{resumen[m]['n_variables']}{extra}</td><td class=l>{ps}</td></tr>"
    return "<table><thead><tr><th>Modelo</th><th>Variables</th><th>Hiperparámetros elegidos</th></tr></thead><tbody>" + filas + "</tbody></table>"


ETIQ = {
    "estimated_delivery_days": "días de entrega prometidos", "global_late_30d": "tasa de retraso del sistema (30 días)",
    "estado_late_30d": "tasa de retraso hacia el estado del cliente (30 días)", "global_n_30d": "entregas del sistema (30 días)",
    "pedidos_ult_7d": "pedidos de los últimos 7 días", "seller_late_rate": "historial de retrasos del vendedor",
    "ruta_late_rate": "historial de retrasos de la ruta", "avg_distance_km": "distancia vendedor–cliente",
    "avg_dias_limite_envio": "plazo del vendedor para despachar", "min_dias_limite_envio": "plazo mínimo para despachar",
    "seller_n_prev": "pedidos previos del vendedor", "ruta_n_prev": "pedidos previos de la ruta",
    "total_freight": "flete total", "freight_ratio": "flete / precio",
}
top_imp = list(res["importancia"].items())[:5]
lista_imp = ", ".join(f"{ETIQ.get(k, k.replace('_', ' '))} ({p(v, 1)})" for k, v in top_imp)

r = eda["reviews"]
T = split["train"]
Te = split["test"]
fig = "../figuras/"

html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Predicción de entregas tardías en Olist</title>
<style>
@page {{ size: A4; margin: 22mm 20mm 22mm 20mm; @bottom-center {{ content: counter(page); }} }}
body {{ font-family: "Times New Roman", Georgia, serif; font-size: 10.6pt; line-height: 1.42; color: #111; margin: 0; background: #fff; }}
.wrap {{ max-width: 170mm; margin: 0 auto; }}
h1 {{ font-size: 18.5pt; line-height: 1.2; text-align: center; margin: 0 0 8pt; font-weight: 700; }}
.autores {{ text-align: center; font-size: 10.5pt; margin-bottom: 2pt; }}
.afil {{ text-align: center; font-size: 9.5pt; color: #444; margin-bottom: 14pt; font-style: italic; }}
.abstract {{ border-top: 1px solid #111; border-bottom: 1px solid #111; padding: 8pt 4mm; margin: 0 0 14pt; font-size: 9.8pt; text-align: justify; }}
.abstract b.h {{ font-variant: small-caps; letter-spacing: .03em; }}
h2 {{ font-size: 12pt; margin: 16pt 0 6pt; font-variant: small-caps; letter-spacing: .02em; }}
h3 {{ font-size: 10.8pt; margin: 10pt 0 4pt; font-style: italic; font-weight: 700; }}
p {{ margin: 0 0 7pt; text-align: justify; hyphens: auto; }}
figure {{ margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }}
figure img {{ max-width: 100%; }}
figure.m img {{ max-width: 78%; }}
figcaption {{ font-size: 9pt; text-align: justify; margin-top: 4pt; color: #222; }}
figcaption b {{ font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 4pt 0 2pt; font-family: "Helvetica Neue", Arial, sans-serif; break-inside: avoid; }}
thead th {{ border-top: 1.4px solid #111; border-bottom: 0.8px solid #111; padding: 4pt 5pt; text-align: right; font-weight: 600; }}
td {{ padding: 3.2pt 5pt; text-align: right; border-bottom: 0.4px solid #ddd; font-variant-numeric: tabular-nums; }}
td:not(:first-child):not(.l) {{ white-space: nowrap; }}
th:first-child, td:first-child, td.l {{ text-align: left; }}
tbody tr:last-child td {{ border-bottom: 1.4px solid #111; }}
tr.win td {{ font-weight: 700; }}
.tcap {{ font-size: 9pt; margin: 10pt 0 3pt; text-align: left; break-after: avoid; }}
ul {{ margin: 0 0 7pt 0; padding-left: 16pt; }} li {{ margin-bottom: 3pt; text-align: justify; }}
.ref p {{ font-size: 9pt; padding-left: 14pt; text-indent: -14pt; margin-bottom: 3pt; text-align: left; }}
code {{ font-size: 9pt; font-family: Menlo, monospace; }}
.kw {{ font-size: 9.4pt; margin-top: 5pt; }}
</style></head><body><div class="wrap">

<h1>Anticipar el retraso: predicción de entregas tardías en un marketplace de e-commerce con validación temporal sin fuga de información</h1>
<div class="autores">Equipo 3</div>
<div class="afil">Diplomado en IA Generativa, Business Intelligence y Ciencia de Datos · Proyecto final · Noviembre 2026</div>

<div class="abstract"><b class="h">Resumen.</b> Las entregas tardías son poco frecuentes en el marketplace brasileño Olist ({p(eda['tasa_tardios'])} de
{n(eda['pedidos'])} pedidos entre 2017 y 2018), pero tienen un costo alto en satisfacción: la calificación promedio cae de
{f(r['score_a_tiempo'], 2)} a {f(r['score_tardio'], 2)} estrellas y el {f(r['pct_1_2_tardio'], 0)} % de los pedidos tardíos recibe una o dos estrellas.
Planteamos un problema de clasificación binaria: estimar, <i>en el momento de la compra</i>, la probabilidad de que un pedido llegue después
de la fecha prometida. Construimos {n_base} variables por pedido que usan exclusivamente información disponible en ese instante, incluidas tasas
históricas <i>point-in-time</i> del vendedor y de la ruta y variables de carga reciente del sistema. Para evaluar sin fuga de información
diseñamos un esquema temporal con un periodo de prueba aislado (mayo–agosto 2018), validación cruzada de ventana creciente de cinco folds y
purga de pedidos cuyo resultado aún no se conocía al inicio de cada periodo de validación. Comparamos regresión logística, Random Forest,
XGBoost y una red neuronal (MLP en Keras), cada uno con su propia selección de variables y búsqueda de hiperparámetros. Los tres modelos
clásicos empataron en validación y la regresión logística los representa por simplicidad; la red neuronal fue la mejor en validación
(PR-AUC {f(cv[NN]['PR-AUC val (prom)'])} frente a {f(cv[mejor]['PR-AUC val (prom)'])}). En el periodo de prueba ambos fueron prácticamente
equivalentes (PR-AUC {f(te[mejor]['PR-AUC'])} y {f(te[NN]['PR-AUC'])}, alrededor de {f(te[mejor]['PR-AUC / azar'], 1)} veces el azar), y la
logística concentró el {f(cap[20]['% tardíos capturados'], 0)} % de los retrasos en el 20 % de pedidos con mayor riesgo. Discutimos el efecto de la
deriva temporal en las métricas y en el umbral de decisión, y proponemos operar el score por capacidad mediante alertas proactivas.
<div class="kw"><b>Palabras clave:</b> clasificación desbalanceada, validación temporal, fuga de información, logística de última milla, e-commerce.</div></div>

<h2>1. Introducción</h2>
<p>En un marketplace, la empresa promete al cliente una fecha de entrega que depende de terceros: el vendedor debe despachar a tiempo y la
transportista debe completar la ruta. Cuando la promesa no se cumple, el costo no es sólo logístico: la reseña pública del pedido se deteriora
y con ella la reputación del vendedor y la conversión de futuros compradores. Si fuera posible identificar con anticipación los pedidos con mayor
riesgo, la empresa podría gestionar la expectativa del cliente o priorizar la operación antes de que el retraso ocurra.</p>
<p>Este trabajo aborda esa pregunta con datos públicos reales de Olist [1]. Sus contribuciones son: (i) un conjunto de variables construidas
estrictamente con información disponible al momento de la compra, verificado contra fugas; (ii) un esquema de validación temporal con purga que
refleja cuándo se conoce realmente la etiqueta de un pedido; (iii) una comparación controlada de cuatro familias de modelos, cada una con su propia
selección de variables y ajuste de hiperparámetros, y (iv) una traducción de los resultados a decisiones operativas.</p>

<h2>2. Datos</h2>
<p>El conjunto de datos de Olist contiene {n(limp['pedidos_originales'])} pedidos (septiembre 2016 – octubre 2018) distribuidos en nueve tablas
relacionales: pedidos, ítems, pagos, productos, vendedores, clientes, geolocalización por código postal y reseñas. Las reglas de limpieza fueron:
eliminar pedidos cancelados o no disponibles ({n(limp['cancelados_o_no_disponibles'])}); excluir pedidos sin fecha de entrega
({n(limp['sin_fecha_de_entrega'])}); y excluir pedidos cuya fecha estimada es posterior a la última entrega registrada, para los cuales aún no se
podría conocer el resultado ({n(limp['censurados_fecha_estimada_posterior'])}). Los pedidos de 2016 son escasos y presentan un hueco entre noviembre
y diciembre; se usaron sólo como historial para las variables, de modo que el conjunto de modelado contiene {n(eda['pedidos'])} pedidos a partir de
enero de 2017. No se eliminaron valores atípicos, porque los pedidos extremos (muy pesados o lejanos) son precisamente los de mayor riesgo.</p>
<p><b>Variable objetivo.</b> Un pedido es tardío (<code>delivery_late = 1</code>) si se entregó en un día posterior a la fecha estimada. La fecha
estimada siempre viene a las 00:00, por lo que la comparación se hace por día: un pedido entregado el mismo día prometido cuenta como a tiempo.
Definimos además el instante en que la etiqueta se vuelve conocida, <code>label_conocida_en = min(entrega, estimada + 1 día)</code>: un pedido
puntual se sabe al entregarse y uno tardío en cuanto vence su promesa. Esta columna es la base de las variables históricas y de la purga.</p>

<figure class="m"><img src="{fig}eda_reviews.png"><figcaption><b>Figura 1.</b> Distribución de calificaciones según puntualidad del pedido.
Los pedidos tardíos promedian {f(r['score_tardio'], 2)} estrellas frente a {f(r['score_a_tiempo'], 2)} de los puntuales; el
{f(r['dist_estrellas']['tardio'][0], 0)} % recibe una estrella. Las reseñas se usan sólo para dimensionar el problema, nunca como variable.</figcaption></figure>

<figure><img src="{fig}eda_tasa_mensual.png"><figcaption><b>Figura 2.</b> Volumen de pedidos y tasa de retraso por mes de compra. Los retrasos llegan en
oleadas (noviembre 2017 y febrero–marzo 2018). La línea punteada marca el inicio del periodo de prueba, que tiene una tasa de retraso menor
({Te['% tardíos']} %) que el periodo de entrenamiento ({T['% tardíos']} %).</figcaption></figure>

<h2>3. Metodología</h2>
<h3>3.1 Variables</h3>
<p>Cada pedido se describe con {n_base} variables agrupadas en ocho familias: promesa de entrega (días prometidos y plazo del vendedor para
despachar), tamaño del pedido, características físicas del producto, forma de pago, geografía (estados, distancia de Haversine entre los centroides
de los códigos postales), calendario (día de la semana y franja horaria), historial y carga reciente. El mes de compra se descartó porque, con menos
de dos años de datos, identifica episodios concretos que el modelo memoriza en lugar de estacionalidad.</p>
<p>Las variables de historial (tasa de retraso del vendedor y de la ruta estado→estado) y de carga reciente (tasa de retraso de las entregas
conocidas en los 30 días previos, global y hacia el estado del cliente, y número de pedidos de los últimos 7 días) son <i>point-in-time</i>: para
un pedido comprado en <i>t</i> sólo cuentan etiquetas con <code>label_conocida_en &lt; t</code>. Las tasas se suavizan hacia un <i>prior</i> con
m = {config.M_SUAVIZADO} observaciones; a diferencia de una versión anterior del proyecto, el prior es la tasa global conocida hasta <i>t</i> y no la del
periodo de entrenamiento completo, que filtraba información futura hacia los primeros folds. Verificamos la ausencia de fuga recalculando por
fuerza bruta el historial de 300 pedidos aleatorios. Las variables categóricas se codificaron con <i>one-hot</i> (las {config.MAX_CATEGORIAS} categorías más
frecuentes en entrenamiento), para un total de {split['n_variables']} columnas.</p>

<h3>3.2 Diseño de validación</h3>
<p>El periodo de prueba son los pedidos comprados a partir del {split['fecha_corte']} ({n(Te['Filas'])} pedidos, {Te['% tardíos']} % tardíos); no participó
en ninguna decisión y se evaluó una sola vez. El entrenamiento ({n(T['Filas'])} pedidos, {T['% tardíos']} % tardíos) se dividió en seis bloques
cronológicos para formar cinco folds de ventana creciente: el fold <i>k</i> entrena con los bloques 1…<i>k</i> y valida con el bloque <i>k</i>+1 (Tabla 1).
A diferencia de <code>TimeSeriesSplit</code>, aplicamos <b>purga</b> [9]: un pedido sólo entrena si su etiqueta ya se conocía al inicio del periodo
de validación. Sin ella, los últimos pedidos del entrenamiento tendrían etiquetas que en producción todavía no existirían. Entre entrenamiento y
prueba se purgaron {n(split['purgados'])} pedidos. La imputación (mediana) y el escalado se ajustan dentro de cada fold mediante <code>Pipeline</code>.</p>
<div class="tcap"><b>Tabla 1.</b> Folds temporales con purga.</div>{tabla_folds()}
<figure><img src="{fig}split_temporal.png"><figcaption><b>Figura 3.</b> Esquema de validación: cinco folds de ventana creciente y periodo de prueba
aislado. La purga retira del entrenamiento los pedidos cuyo resultado aún no se conocía al inicio de cada validación (huecos apenas visibles a esta escala).</figcaption></figure>

<h3>3.3 Modelos, selección de variables e hiperparámetros</h3>
<p>Se compararon cuatro modelos: regresión logística con pesos balanceados (línea base), Random Forest [3], XGBoost [4] y un perceptrón multicapa
en Keras [5]; los modelos clásicos se implementaron con scikit-learn [2]. Para cada modelo clásico se siguió la misma secuencia: (1) eliminación recursiva de variables con validación cruzada (RFECV [7]),
eliminando el 10 % de variables menos importantes por ronda según los coeficientes o la importancia propia del modelo; (2) búsqueda aleatoria de
hiperparámetros [8] sobre las variables elegidas. Ambos pasos usan los mismos cinco folds y la métrica PR-AUC. Como una red no ofrece una medida
de importancia directa, su selección de variables comparó conjuntos candidatos (todas, el RFECV de cada modelo clásico, su unión y su
intersección), seguida de una búsqueda sobre arquitectura, <i>dropout</i>, regularización L2, tasa de aprendizaje y tamaño de lote. El
<i>early stopping</i> de la red usa la cola más reciente (15 %) del entrenamiento de cada fold, nunca el fold de validación.</p>

<h3>3.4 Métricas, umbral y regla de selección</h3>
<p>Con {p(eda['tasa_tardios'])} de positivos, la exactitud no es informativa: predecir siempre «a tiempo» acierta más del 93 %. La métrica principal es
el área bajo la curva precisión-recall (PR-AUC, <i>average precision</i>) [6], cuyo valor de azar es la tasa de positivos; reportamos también
ROC-AUC y precisión, recall y F1 de la clase tardía. El sobreajuste se diagnostica con la brecha de ROC-AUC entre entrenamiento y validación,
porque el PR-AUC depende de la tasa de positivos y ésta cambia entre periodos. El umbral de decisión de cada modelo es el que maximiza el F1 sobre
sus predicciones fuera de fold. La regla de selección se fijó antes de ver resultados: mayor PR-AUC promedio en validación y, en caso de empate
(diferencia menor que {f(config.TOLERANCIA_EMPATE, 3)}), el modelo más simple (logística, Random Forest, XGBoost, red neuronal).</p>

<h2>4. Resultados</h2>
<h3>4.1 Comparación en validación temporal</h3>
<p>La Tabla 2 resume el desempeño en los cinco folds. {frase_empate} {frase_nn} (PR-AUC {f(cv[NN]['PR-AUC val (prom)'])} frente a
{f(cv[mejor]['PR-AUC val (prom)'])}). La desviación entre folds es grande para todos los modelos y sigue a la tasa de retraso de cada periodo
(Figura 4): los folds con oleadas de retraso son más fáciles de ordenar, y el primero, con menos de 3 % de positivos y sólo cinco meses de historia,
es el más difícil.</p>
<div class="tcap"><b>Tabla 2.</b> Validación cruzada temporal (promedio de 5 folds). «× azar» = PR-AUC / tasa de tardíos fuera de fold ({p(res['azar_cv'], 2)}).</div>{tabla_cv()}
<figure><img src="{fig}comparacion_cv.png"><figcaption><b>Figura 4.</b> Izquierda: PR-AUC de validación (promedio ± desviación). Centro: ROC-AUC de
entrenamiento (gris) y validación (color); la brecha indica sobreajuste. Derecha: curvas precisión-recall de las predicciones fuera de fold.</figcaption></figure>
<div class="tcap"><b>Tabla 3.</b> Variables e hiperparámetros elegidos.</div>{tabla_params()}

<h3>4.2 Prueba final</h3>
<p>El periodo de prueba se abrió una sola vez para el mejor modelo clásico y la red neuronal (Tabla 4). {frase_test} Los valores absolutos son menores que en validación porque el periodo de prueba tiene menos retrasos ({p(res['azar_test'])}); la mejora
relativa frente al azar es del orden de {f(te[mejor]['PR-AUC / azar'], 1)} veces para {mejor} y {f(te[NN]['PR-AUC / azar'], 1)} para la red. En términos
operativos (Tabla 5, Figura 5), revisar el 10 % de pedidos con mayor riesgo permite anticipar el {f(cap[10]['% tardíos capturados'], 0)} % de los
retrasos, y el 20 % de pedidos concentra el {f(cap[20]['% tardíos capturados'], 0)} %.</p>
<p>El umbral fijo de probabilidad elegido en validación no se transfirió bien: en prueba marca el {f(marcados * 100, 0)} % de los pedidos, con
recall alto ({p(te[mejor]['Recall'], 0)}) pero precisión baja ({p(te[mejor]['Precision'], 0)}). Esto ocurre porque la tasa de retraso bajó y porque
el umbral se eligió sobre predicciones de cinco modelos de fold con escalas de probabilidad distintas. La métrica independiente del umbral
(PR-AUC) y la operación por capacidad (revisar el <i>k</i> % de mayor riesgo) son robustas a esa deriva.</p>
<div class="tcap"><b>Tabla 4.</b> Desempeño en el periodo de prueba ({n(res['n_test'])} pedidos, {n(res['tardios_test'])} tardíos). El umbral se eligió en validación.</div>{tabla_test()}
<figure><img src="{fig}test_final.png"><figcaption><b>Figura 5.</b> Matrices de confusión y curvas precisión-recall en el periodo de prueba.</figcaption></figure>
<div class="tcap"><b>Tabla 5.</b> Captura de retrasos al revisar el <i>k</i> % de pedidos con mayor riesgo (prueba).</div>{tabla_captura()}

<h3>4.3 Variables relevantes</h3>
<p>Las variables con mayor peso en {mejor} son: {lista_imp}. El patrón es coherente con el análisis exploratorio: la promesa de entrega, la
saturación reciente del sistema y el historial de la ruta y del vendedor explican la mayor parte del riesgo.</p>
<figure class="m"><img src="{fig}importancia_final.png"><figcaption><b>Figura 6.</b> Importancia relativa de las variables del mejor modelo clásico.</figcaption></figure>

<h2>5. Discusión</h2>
<p><b>Implicaciones.</b> El modelo no elimina el retraso, pero ordena los pedidos por riesgo con una calidad de alrededor de {f(te[mejor]['PR-AUC / azar'], 1)} veces el azar, lo que
permite concentrar recursos limitados. El umbral es una palanca de negocio: para acciones baratas, como avisar al cliente, conviene un umbral bajo
(más recall); para intervenciones costosas, uno alto (más precisión). Proponemos un piloto de alertas proactivas sobre el 10–20 % de pedidos de
mayor riesgo, medido con un experimento A/B sobre la proporción de reseñas negativas.</p>
<p><b>Limitaciones.</b> (i) Deriva temporal: la tasa de retraso varía de 1 % a 19 % entre meses, lo que se refleja en la alta variabilidad entre
folds y en la diferencia entre validación y prueba. (ii) Faltan variables exógenas determinantes, como transportista, clima, huelgas o festivos.
(iii) La fecha prometida por Olist ya incorpora un margen (el pedido típico llega {f(eda['dias_antes_mediana'], 0)} días antes), por lo que los retrasos
restantes son, en buena medida, eventos difíciles de anticipar. (iv) Un umbral fijo de probabilidad es frágil ante la deriva; recomendamos fijar la capacidad de revisión o recalibrar periódicamente.
(v) Los mismos folds se usaron para elegir variables, hiperparámetros y umbral, por
lo que las métricas de validación son optimistas; por eso reportamos un periodo de prueba aislado. (vi) En iteraciones anteriores del proyecto se
consultó el periodo de prueba; la versión final fijó todas las decisiones con validación cruzada antes de evaluarlo.</p>

<h2>6. Conclusiones</h2>
<p>Con información disponible al momento de la compra es posible anticipar una parte relevante de las entregas tardías de Olist. El aporte
principal no es la complejidad del modelo, sino el diseño de evaluación: variables <i>point-in-time</i>, validación temporal con purga y una regla de
selección fijada de antemano. {frase_conclusion} Como trabajo futuro proponemos incorporar fuentes exógenas, calibrar las probabilidades para fijar umbrales por costo, modelar
también los días de retraso y validar el impacto en un experimento controlado.</p>
<p><b>Reproducibilidad.</b> El código está organizado en notebooks secuenciales (<code>01_limpieza</code> a <code>06_seleccion_y_test_final</code>) y
módulos en <code>src/</code>; todos los resultados de este documento se generan automáticamente a partir de sus salidas.</p>

<h2>Referencias</h2>
<div class="ref">
<p>[1] Olist. <i>Brazilian E-Commerce Public Dataset by Olist</i>. Kaggle, 2018. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce</p>
<p>[2] Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830, 2011.</p>
<p>[3] Breiman, L. Random Forests. <i>Machine Learning</i>, 45(1), 5–32, 2001.</p>
<p>[4] Chen, T. y Guestrin, C. XGBoost: A Scalable Tree Boosting System. <i>Proceedings of KDD</i>, 785–794, 2016.</p>
<p>[5] Chollet, F. et al. Keras. https://keras.io, 2015.</p>
<p>[6] Saito, T. y Rehmsmeier, M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. <i>PLoS ONE</i>, 10(3), 2015.</p>
<p>[7] Guyon, I., Weston, J., Barnhill, S. y Vapnik, V. Gene Selection for Cancer Classification using Support Vector Machines. <i>Machine Learning</i>, 46, 389–422, 2002.</p>
<p>[8] Bergstra, J. y Bengio, Y. Random Search for Hyper-Parameter Optimization. <i>Journal of Machine Learning Research</i>, 13, 281–305, 2012.</p>
<p>[9] López de Prado, M. <i>Advances in Financial Machine Learning</i>, cap. 7: Cross-Validation in Finance. Wiley, 2018.</p>
<p>[10] Bergmeir, C. y Benítez, J. M. On the use of cross-validation for time series predictor evaluation. <i>Information Sciences</i>, 191, 192–213, 2012.</p>
</div>
</div></body></html>"""

carpeta = RAIZ / "reportes" / "articulo"
carpeta.mkdir(parents=True, exist_ok=True)
salida = carpeta / "articulo.html"
salida.write_text(html)
print("HTML:", salida)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
pdf = carpeta / "reporte_tecnico.pdf"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                f"--print-to-pdf={pdf}", salida.as_uri()], check=True, capture_output=True)
print("PDF:", pdf)
