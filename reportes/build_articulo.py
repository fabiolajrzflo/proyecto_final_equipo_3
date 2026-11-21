"""Genera el reporte técnico (artículo científico corto) en HTML y PDF con los resultados reales.

Sigue, en orden y con la misma numeración, las 8 secciones de la rúbrica del proyecto final
(Resumen · Introducción y definición del problema · Estado del arte · Datos · Metodología ·
Resultados · Retos, límites y conclusión · Referencias en APA 7) más el diccionario de datos en un anexo.

Uso (desde la raíz del proyecto):  .venv/bin/python reportes/build_articulo.py
Requiere haber corrido los notebooks 01–07.
"""
import json
import re
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
comp = json.loads((M / "complementos.json").read_text())
params = {n: json.loads((config.MODELOS / c / "best_params.json").read_text()) for n, c in config.CLAVES_MODELO.items()}
resumen = res["resumen_modelos"]
cols = pd.read_parquet(config.DATOS_PROCESSED / "dataset_pedidos.parquet").columns
n_base = sum(c not in COLUMNAS_EXCLUIR for c in cols)

NN = config.NOMBRE_RED
mejor = res["mejor_clasico"]
cv = {r["Modelo"]: r for r in res["comparacion_cv"]}
te = {r["Modelo"]: r for r in res["test"]}
cap = {int(round(r["% pedidos revisados"])): r for r in res["captura"][mejor]}
cap_nn = {int(round(r["% pedidos revisados"])): r for r in res["captura"][NN]}
equil = {int(r["% revisado"]): r for r in comp["equilibrio"]}



def f(x, d=3):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", " ")


def p(x, d=1):
    return f(x * 100, d) + " %"


def n(x):
    return f"{int(round(float(x))):,}".replace(",", " ")


def pv(x):
    return "&lt; 0,001" if x < 0.001 else f(x, 2)


# ---------------------------------------------------------------- frases que dependen de los resultados
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
paridad = abs(te[NN]["PR-AUC"] - te[mejor]["PR-AUC"]) < 0.01
conf = res["confusion"][mejor]
marcados = (conf["tp"] + conf["fp"]) / sum(conf.values())
frase_test = (
    f"En prueba ambos modelos resultaron prácticamente equivalentes (PR-AUC {f(te[mejor]['PR-AUC'])} y {f(te[NN]['PR-AUC'])}; "
    f"ROC-AUC {f(te[mejor]['ROC-AUC'])} y {f(te[NN]['ROC-AUC'])}): la ventaja de la red en validación no se sostuvo en el futuro."
    if paridad else f"{mejor_test} obtuvo el mayor PR-AUC en prueba.")

r = eda["reviews"]
rv = comp["reviews"]
T, Te = split["train"], split["test"]
tasas = comp["tasas"]
carga = comp["carga"]["tabla"]
rec = comp["recompra"]
cr = comp["costo_resenas"]
val = comp["valor"]
elegida, posthoc = comp["regla_elegida"], comp["regla_posthoc"]
dias_test = (pd.Timestamp(Te["Hasta"]) - pd.Timestamp(Te["Desde"])).days + 1
ETIQ = {
    "estimated_delivery_days": "días de entrega prometidos", "global_late_30d": "tasa de retraso del sistema (30 días)",
    "estado_late_30d": "tasa de retraso hacia el estado del cliente (30 días)", "global_n_30d": "entregas del sistema (30 días)",
    "pedidos_ult_7d": "pedidos de los últimos 7 días", "seller_late_rate": "historial de retrasos del vendedor",
    "ruta_late_rate": "historial de retrasos de la ruta", "avg_distance_km": "distancia vendedor–cliente",
    "avg_dias_limite_envio": "plazo del vendedor para despachar", "min_dias_limite_envio": "plazo mínimo para despachar",
    "seller_n_prev": "pedidos previos del vendedor", "ruta_n_prev": "pedidos previos de la ruta",
    "total_freight": "flete total", "freight_ratio": "flete / precio",
    "same_state": "vendedor y cliente en el mismo estado",
}
lista_imp = ", ".join(f"{ETIQ.get(k, k.replace('_', ' '))} ({p(v, 1)})" for k, v in list(res["importancia"].items())[:5])
estados = eda["estados_train"]


# ---------------------------------------------------------------- tablas
def tabla(cabecera, filas, clase=""):
    th = "".join(f"<th>{c}</th>" for c in cabecera)
    return f"<table class='{clase}'><thead><tr>{th}</tr></thead><tbody>{''.join(filas)}</tbody></table>"


def tabla_costos():
    e20 = equil[20]
    filas = [
        "<tr><td rowspan=3 class=l><b>Falso negativo</b><br>retraso no anticipado</td>"
        "<td class=l>La reseña se deteriora</td><td class=l>Diferencia en reseñas de 1–2★, tardíos vs. puntuales</td>"
        f"<td class=l>+{f(rv['dp'] * 100, 1)} pp (IC 95 %: {f(rv['dp_li'] * 100, 1)}–{f(rv['dp_ls'] * 100, 1)}); "
        f"≈ {n(cr['resenas_extra'])} reseñas negativas adicionales en los cuatro meses de prueba ({n(cr['por_mil_pedidos'])} por cada 1 000 pedidos)</td></tr>",
        "<tr><td class=l>Valor comprometido</td><td class=l>Pago total de los pedidos tardíos en prueba</td>"
        f"<td class=l>R$ {n(val['gmv_tardios_test'])} (ticket promedio R$ {n(val['ticket_tardio'])})</td></tr>",
        "<tr><td class=l>Pérdida del cliente</td><td class=l>Recompra en los 180 días siguientes</td>"
        f"<td class=l>Sin diferencia significativa ({p(rec['tardio']['p'], 1)} vs. {p(rec['a_tiempo']['p'], 1)}; p = {pv(rec['p'])})</td></tr>",
        "<tr><td class=l><b>Falso positivo</b><br>alerta innecesaria</td><td class=l>Costo de contactar o intervenir un pedido que llegaría a tiempo</td>"
        f"<td class=l>Alertas por cada retraso anticipado (revisando el 20 %)</td><td class=l>{f(e20['Alertas por retraso anticipado'], 1)} alertas: "
        f"la acción conviene si gestionar un retraso vale más que {f(e20['Alertas por retraso anticipado'], 1)} alertas</td></tr>",
    ]
    return tabla(["Error", "Qué ocurre", "Cómo se mide", "Estimación con los datos"], filas, "costos")


def tabla_arte():
    filas = [
        "<tr><td class=l>Salari, Liu y Shen (2022)</td><td class=l><i>MSOM</i> (INFORMS), datos de JD.com</td>"
        "<td class=l>Pronosticar el tiempo de entrega y fijar la promesa en retail en línea</td>"
        "<td class=l>Modelos de árboles que estiman la distribución del tiempo de entrega; la nueva política de promesa eleva las ventas 6,1 %</td>"
        "<td class=l>La promesa es una palanca de negocio. Aquí la promesa ya está dada y se predice su incumplimiento</td></tr>",
        "<tr><td class=l>Derrow-Pinion et al. (2021)</td><td class=l>CIKM, Google y DeepMind</td>"
        "<td class=l>Tiempo estimado de llegada (ETA) en Google Maps</td>"
        "<td class=l>Redes neuronales de grafos sobre la red vial; hasta 40 % menos errores graves de ETA en ciudades como Sídney</td>"
        "<td class=l>Usan tráfico en tiempo real que este caso no tiene: sólo hay información del momento de la compra</td></tr>",
        "<tr><td class=l>Hu et al. (2022)</td><td class=l>arXiv, Uber</td>"
        "<td class=l>Corregir el ETA de un motor de rutas</td>"
        "<td class=l>Modelo híbrido que aprende el residuo sobre una predicción base</td>"
        "<td class=l>Partir de una estimación existente; análogo a tratar la fecha prometida como base</td></tr>",
        "<tr><td class=l>Grinsztajn, Oyallon y Varoquaux (2022)</td><td class=l>NeurIPS, Inria</td>"
        "<td class=l>¿Árboles o redes en datos tabulares?</td>"
        "<td class=l>45 conjuntos de datos: los árboles siguen siendo superiores en tablas medianas (~10 000 filas)</td>"
        "<td class=l>Justifica comparar modelos de árboles contra la red y no esperar una ventaja grande de la red</td></tr>",
        "<tr><td class=l>Kaufman et al. (2012)</td><td class=l><i>ACM TKDD</i></td>"
        "<td class=l>Fuga de información en minería de datos</td>"
        "<td class=l>Formalizan la fuga: usar información que no estaría disponible al predecir</td>"
        "<td class=l>Base de las variables <i>point-in-time</i> y de la purga entre periodos</td></tr>",
    ]
    return tabla(["Trabajo", "Publicación / origen", "Problema", "Enfoque y hallazgo", "Relación con este trabajo"], filas, "arte")


def fmt_desc(col, v):
    if "late" in col:
        return f(v * 100, 1) + " %"
    if col == "pedidos_ult_7d":
        return n(v)
    return f(v, 1)


def tabla_descriptiva():
    filas = [f"<tr><td class=l>{d['Variable']}</td><td>{fmt_desc(d['variable'], d['Mediana a tiempo'])}</td>"
             f"<td>{fmt_desc(d['variable'], d['Mediana tardío'])}</td><td>{fmt_desc(d['variable'], d['Media a tiempo'])}</td>"
             f"<td>{fmt_desc(d['variable'], d['Media tardío'])}</td><td>{f(d['AUC univariado'], 3)}</td></tr>"
             for d in sorted(comp["descriptiva"], key=lambda d: -abs(d["AUC univariado"] - 0.5))]
    return tabla(["Variable", "Mediana a tiempo", "Mediana tardío", "Media a tiempo", "Media tardío", "AUC univariado"], filas)


def tabla_inferencia():
    q1, q4 = carga[0], carga[-1]
    filas = [
        f"<tr><td class=l>Tasa de retraso, 2017–2018</td><td>{p(tasas['global']['p'], 2)}</td>"
        f"<td>{f(tasas['global']['li'] * 100, 2)}–{f(tasas['global']['ls'] * 100, 2)} %</td><td class=l>Intervalo de Wilson</td><td>—</td></tr>",
        f"<tr><td class=l>Cambio entre periodos: entrenamiento vs. prueba</td><td>{p(tasas['train']['p'], 2)} vs. {p(tasas['test']['p'], 2)}</td>"
        f"<td>{f(tasas['train']['li'] * 100, 2)}–{f(tasas['train']['ls'] * 100, 2)} % / {f(tasas['test']['li'] * 100, 2)}–{f(tasas['test']['ls'] * 100, 2)} %</td>"
        f"<td class=l>χ² = {f(comp['drift']['chi2'], 1)}</td><td>{pv(comp['drift']['p'])}</td></tr>",
        f"<tr><td class=l>Calificación promedio, tardío − a tiempo</td><td>{f(rv['dif'], 2)} ★</td>"
        f"<td>{f(rv['dif_li'], 2)} a {f(rv['dif_ls'], 2)}</td><td class=l>t de Welch; Mann-Whitney</td><td>{pv(rv['p_welch'])}; {pv(rv['p_mw'])}</td></tr>",
        f"<tr><td class=l>Reseñas de 1–2★, tardío − a tiempo</td><td>+{f(rv['dp'] * 100, 1)} pp</td>"
        f"<td>{f(rv['dp_li'] * 100, 1)} a {f(rv['dp_ls'] * 100, 1)} pp</td><td class=l>χ²</td><td>{pv(rv['p_chi'])}</td></tr>",
        f"<tr><td class=l>Retraso según la carga reciente hacia el estado (cuartil 1 vs. 4)</td><td>{p(q1['% tardíos'], 1)} vs. {p(q4['% tardíos'], 1)}</td>"
        f"<td>{f(q1['IC inf'] * 100, 1)}–{f(q1['IC sup'] * 100, 1)} % / {f(q4['IC inf'] * 100, 1)}–{f(q4['IC sup'] * 100, 1)} %</td>"
        f"<td class=l>χ² = {n(comp['carga']['chi2'])} (4 cuartiles)</td><td>{pv(comp['carga']['p'])}</td></tr>",
        f"<tr><td class=l>Recompra a 180 días, primera compra tardía vs. puntual</td><td>{p(rec['tardio']['p'], 2)} vs. {p(rec['a_tiempo']['p'], 2)}</td>"
        f"<td>{f(rec['tardio']['li'] * 100, 2)}–{f(rec['tardio']['ls'] * 100, 2)} % / {f(rec['a_tiempo']['li'] * 100, 2)}–{f(rec['a_tiempo']['ls'] * 100, 2)} %</td>"
        f"<td class=l>χ² = {f(rec['chi2'], 2)}</td><td>{pv(rec['p'])}</td></tr>",
    ]
    return tabla(["Hallazgo", "Estimación", "IC 95 %", "Prueba", "p-valor"], filas)


def tabla_folds():
    filas = [f"<tr><td>{r['Fold']}</td><td>{r['Train desde'][:10]} – {r['Train hasta'][:10]}</td><td>{n(r['Filas train'])}</td>"
             f"<td>{r['Val desde'][:10]} – {r['Val hasta'][:10]}</td><td>{n(r['Filas val'])}</td>"
             f"<td>{f(r['% tardíos train'], 2)}</td><td>{f(r['% tardíos val'], 2)}</td></tr>" for r in split["folds"]]
    return tabla(["Fold", "Periodo de entrenamiento", "Filas", "Periodo de validación", "Filas", "% tardíos train", "% tardíos val"], filas)


def tabla_cv():
    folds = {m: pd.DataFrame(v) for m, v in res["cv_por_fold"].items()}   # precisión y recall con el umbral fuera de fold
    filas = [f"<tr{' class=win' if m == mejor else ''}><td>{m}</td><td>{int(cv[m]['Variables'])}</td>"
             f"<td>{f(cv[m]['PR-AUC val (prom)'])} ± {f(cv[m]['PR-AUC val (desv)'])}</td><td>{f(cv[m]['PR-AUC val / azar'], 2)}</td>"
             f"<td>{f(cv[m]['ROC-AUC val (prom)'])}</td><td>{f(cv[m]['ROC-AUC train (prom)'])}</td><td>{f(cv[m]['Brecha ROC train-val'])}</td>"
             f"<td>{f(folds[m]['Precision val'].mean())}</td><td>{f(folds[m]['Recall val'].mean())}</td><td>{f(cv[m]['F1 fuera de fold'])}</td></tr>"
             for m in orden]
    return tabla(["Modelo", "Variables", "PR-AUC val", "× azar", "ROC-AUC val", "ROC-AUC train", "Brecha ROC",
                  "Precisión", "Recall", "F1 (OOF)"], filas)


def fmt_param(v):
    if isinstance(v, float):
        return f"{v:.4g}".replace(".", ",") if abs(v) < 1 else f(v, 2)
    if isinstance(v, list):
        return "(" + ", ".join(map(str, v)) + ")"
    return str(v)


def tabla_params():
    filas = []
    for m in config.CLAVES_MODELO:
        ps = "; ".join(f"{k.replace('modelo__', '')} = {fmt_param(v)}" for k, v in params[m].items())
        extra = f" (conjunto: {resumen[m].get('conjunto_variables')})" if m == NN else ""
        filas.append(f"<tr><td>{m}</td><td class=l>{resumen[m]['n_variables']}{extra}</td><td class=l>{ps}</td></tr>")
    return tabla(["Modelo", "Variables", "Hiperparámetros elegidos"], filas)


def tabla_test():
    filas = [f"<tr{' class=win' if m == mejor_test else ''}><td>{m}</td><td>{f(te[m]['PR-AUC CV'])}</td><td>{f(te[m]['PR-AUC'])}</td>"
             f"<td>{f(te[m]['PR-AUC / azar'], 2)}</td><td>{f(te[m]['ROC-AUC'])}</td><td>{f(te[m]['Precision'])}</td>"
             f"<td>{f(te[m]['Recall'])}</td><td>{f(te[m]['F1'])}</td><td>{f(te[m]['Umbral'], 2)}</td></tr>" for m in [mejor, NN]]
    return tabla(["Modelo", "PR-AUC CV", "PR-AUC prueba", "× azar", "ROC-AUC", "Precisión", "Recall", "F1", "Umbral"], filas)


def tabla_capacidad():
    filas = [f"<tr><td>{k} %</td><td>{n(equil[k]['Alertas'])}</td><td>{f(equil[k]['Alertas'] / dias_test, 0)}</td>"
             f"<td>{n(equil[k]['Tardíos anticipados'])} ({f(cap[k]['% tardíos capturados'], 1)} %)</td>"
             f"<td>{f(cap[k]['Precision'] * 100, 1)} %</td><td>{f(cap[k]['Lift'], 2)}</td>"
             f"<td>{f(equil[k]['Alertas por retraso anticipado'], 1)}</td><td>{f(cap_nn[k]['% tardíos capturados'], 1)} %</td></tr>"
             for k in sorted(cap)]
    return tabla(["Pedidos revisados", "Alertas", "Alertas / día", f"Tardíos anticipados ({mejor})", "Precisión", "Lift",
                  "Alertas por retraso anticipado", "Captura de la red"], filas)


def tabla_baselines():
    filas = []
    for b in comp["baselines"]:
        marca = " ◆" if b["Método"] == elegida["Método"] else (" ◇" if b["Método"] == posthoc["Método"] else "")
        ptr = "—" if b.get("PR-AUC train") is None or b["Tipo"] != "regla" else f(b["PR-AUC train"])
        clase = " class=win" if b["Método"] == mejor else ""
        filas.append(f"<tr{clase}><td>{b['Método']}{marca}</td><td>{ptr}</td><td>{f(b['PR-AUC'])}</td><td>{f(b['× azar'], 2)}</td>"
                     f"<td>{f(b['ROC-AUC'])}</td><td>{f(b['Captura top 20 %'] * 100, 1)} %</td></tr>")
    return tabla(["Método", "PR-AUC train", "PR-AUC prueba", "× azar", "ROC-AUC", "Captura top 20 %"], filas)


DICCIONARIO = [
    # (variable, tipo, descripción)
    ("delivery_late", "binaria (objetivo)", "1 si el pedido se entregó en un día posterior a la fecha estimada; 0 si llegó ese día o antes."),
    ("label_conocida_en", "fecha (no entra al modelo)", "Instante en que se conoce el resultado: min(entrega, fecha estimada + 1 día). Base de las variables históricas y de la purga."),
    ("purchase_weekday", "categórica", "Día de la semana de la compra (0 = lunes … 6 = domingo)."),
    ("franja_horaria", "categórica", "Franja de la hora de compra: madrugada (0–5 h), mañana (6–11), tarde (12–17), noche (18–23)."),
    ("estimated_delivery_days", "numérica", "Días entre la compra y la fecha de entrega prometida."),
    ("customer_state", "categórica", "Estado del cliente."),
    ("seller_state", "categórica", "Estado del vendedor principal (primer ítem del pedido)."),
    ("n_seller_states", "numérica", "Número de estados distintos de los vendedores del pedido."),
    ("same_state", "binaria", "1 si vendedor principal y cliente están en el mismo estado."),
    ("avg_distance_km, max_distance_km", "numérica", "Distancia de Haversine media y máxima entre vendedor y cliente, con el centroide de cada código postal."),
    ("total_items, total_products, total_sellers", "numérica", "Número de ítems, de productos distintos y de vendedores distintos del pedido."),
    ("unique_product_categories", "numérica", "Número de categorías de producto distintas."),
    ("main_category", "categórica", "Categoría del primer producto del pedido."),
    ("total_price, total_freight", "numérica", "Suma del precio y del flete de los ítems (R$)."),
    ("freight_ratio", "numérica", "Flete total / precio total."),
    ("avg_product_weight, max_product_weight", "numérica", "Peso medio y máximo de los productos (g)."),
    ("avg_product_length, avg_product_height, avg_product_width", "numérica", "Dimensiones medias de los productos (cm)."),
    ("avg_dias_limite_envio, min_dias_limite_envio", "numérica", "Días entre la compra y la fecha límite para que el vendedor despache (media y mínimo de los ítems)."),
    ("total_payment", "numérica", "Suma de los pagos del pedido (R$)."),
    ("max_installments", "numérica", "Máximo número de mensualidades."),
    ("payment_types", "numérica", "Número de medios de pago distintos."),
    ("main_payment_type", "categórica", "Medio de pago más usado en el pedido."),
    ("seller_late_rate, seller_n_prev", "numérica", f"Tasa de retraso suavizada (m = {config.M_SUAVIZADO}) y número de pedidos previos del vendedor, sólo con resultados conocidos antes de la compra."),
    ("ruta_late_rate, ruta_n_prev", "numérica", "Igual que la anterior, para la ruta estado del vendedor → estado del cliente."),
    ("global_late_30d, global_n_30d", "numérica", f"Tasa de retraso suavizada y número de entregas cuyo resultado se conoció en los {config.DIAS_VENTANA} días previos a la compra (todo el sistema)."),
    ("estado_late_30d", "numérica", f"Tasa de retraso suavizada de las entregas hacia el estado del cliente conocidas en los {config.DIAS_VENTANA} días previos."),
    ("pedidos_ult_7d", "numérica", "Número de compras registradas en la plataforma en los 7 días previos (incluye canceladas)."),
]


def tabla_diccionario():
    filas = [f"<tr><td class='l var'>{v.replace(', ', ',<br>')}</td><td class=l>{t}</td><td class=l>{d}</td></tr>" for v, t, d in DICCIONARIO]
    return tabla(["Variable", "Tipo", "Descripción"], filas, "dicc")


# ---------------------------------------------------------------- resumen (150–250 palabras, lo pide la rúbrica)
resumen_txt = f"""Las entregas tardías son poco frecuentes en el marketplace brasileño Olist ({p(eda['tasa_tardios'])} de {n(eda['pedidos'])}
pedidos entre 2017 y 2018), pero cuestan caro: un pedido tardío tiene {f(rv['dp'] * 100, 1)} puntos porcentuales más probabilidad de
recibir una reseña de una o dos estrellas (IC 95 %: {f(rv['dp_li'] * 100, 1)}–{f(rv['dp_ls'] * 100, 1)}). Este trabajo apoya la decisión
diaria del área de operaciones sobre qué pedidos atender de forma proactiva. Planteamos una clasificación binaria que estima, en el
momento de la compra, la probabilidad de retraso con {n_base} variables disponibles en ese instante, incluidas tasas históricas
<i>point-in-time</i> del vendedor y de la ruta y la carga reciente del sistema. La evaluación usa un periodo de prueba aislado
(mayo–agosto 2018), validación cruzada temporal de cinco folds con purga y la PR-AUC como métrica principal. Comparamos regresión
logística, Random Forest y XGBoost con una red neuronal en Keras con detención temprana. Los modelos clásicos empataron en validación
y la red fue la mejor (PR-AUC {f(cv[NN]['PR-AUC val (prom)'])} frente a {f(cv[mejor]['PR-AUC val (prom)'])}), pero en prueba ambos
resultaron equivalentes ({f(te[mejor]['PR-AUC'])} y {f(te[NN]['PR-AUC'])}, {f(te[mejor]['PR-AUC / azar'], 1)} veces el azar) y superaron
a la regla simple elegida con datos de entrenamiento ({f(elegida['PR-AUC'])}). Revisar el 20 % de pedidos más riesgosos anticipa el
{f(cap[20]['% tardíos capturados'], 0)} % de los retrasos. Recomendamos un piloto de alertas proactivas por capacidad con la regresión
logística, la red como modelo retador y reentrenamiento mensual."""
palabras = len(re.sub(r"<[^>]+>", " ", resumen_txt).split())
assert 150 <= palabras <= 250, f"El resumen tiene {palabras} palabras (la rúbrica pide 150–250)"

fig = "../figuras/"
mejor_corto = "la regresión logística" if mejor == "Regresión Logística" else mejor
autores = "Fabiola Juárez Flores · Martha Jessica Campos Aldape · Candy del Carmen Cortes Castro"

html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Predicción de entregas tardías en Olist · Equipo 3</title>
<style>
@page {{ size: A4; margin: 22mm 20mm 22mm 20mm; @bottom-center {{ content: counter(page); font-size: 9pt; }} }}
body {{ font-family: "Times New Roman", Georgia, serif; font-size: 10.6pt; line-height: 1.42; color: #111; margin: 0; background: #fff; }}
.wrap {{ max-width: 170mm; margin: 0 auto; }}
h1 {{ font-size: 18.5pt; line-height: 1.2; text-align: center; margin: 0 0 8pt; font-weight: 700; }}
.autores {{ text-align: center; font-size: 10.5pt; margin-bottom: 2pt; }}
.afil {{ text-align: center; font-size: 9.5pt; color: #444; margin-bottom: 14pt; font-style: italic; line-height: 1.35; }}
.abstract {{ border-top: 1px solid #111; border-bottom: 1px solid #111; padding: 8pt 4mm; margin: 0 0 14pt; font-size: 9.8pt; text-align: justify; }}
.abstract h2 {{ margin: 0 0 4pt; font-size: 10.5pt; }}
h2 {{ font-size: 12pt; margin: 16pt 0 6pt; font-variant: small-caps; letter-spacing: .02em; break-after: avoid; }}
h3 {{ font-size: 10.8pt; margin: 10pt 0 4pt; font-style: italic; font-weight: 700; break-after: avoid; }}
p {{ margin: 0 0 7pt; text-align: justify; hyphens: auto; }}
figure {{ margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }}
figure img {{ max-width: 100%; }}
figure.m img {{ max-width: 78%; }}
figcaption {{ font-size: 9pt; text-align: justify; margin-top: 4pt; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 4pt 0 2pt; font-family: "Helvetica Neue", Arial, sans-serif; break-inside: avoid; }}
thead th {{ border-top: 1.4px solid #111; border-bottom: 0.8px solid #111; padding: 4pt 5pt; text-align: right; font-weight: 600; vertical-align: bottom; }}
td {{ padding: 3.2pt 5pt; text-align: right; border-bottom: 0.4px solid #ddd; font-variant-numeric: tabular-nums; vertical-align: top; }}
td:not(:first-child):not(.l) {{ white-space: nowrap; }}
th:first-child, td:first-child, td.l {{ text-align: left; }}
tbody tr:last-child td {{ border-bottom: 1.4px solid #111; }}
tr.win td {{ font-weight: 700; }}
table.costos td, table.arte td {{ font-size: 8.2pt; line-height: 1.3; }}
table.arte th, table.costos th {{ text-align: left; }}
table.dicc {{ font-size: 8pt; break-inside: auto; }} table.dicc tr {{ break-inside: avoid; }} table.dicc th {{ text-align: left; }}
table.dicc td.var {{ font-family: Menlo, monospace; font-size: 7.4pt; white-space: nowrap; }}
.tcap {{ font-size: 9pt; margin: 10pt 0 3pt; text-align: left; break-after: avoid; }}
ul, ol {{ margin: 0 0 7pt 0; padding-left: 16pt; }} li {{ margin-bottom: 3pt; text-align: justify; }}
.ref p {{ font-size: 8.8pt; line-height: 1.35; padding-left: 14pt; text-indent: -14pt; margin-bottom: 3pt; text-align: left; overflow-wrap: anywhere; }}
code {{ font-size: 8.6pt; font-family: Menlo, monospace; }}
.kw {{ font-size: 9.4pt; margin-top: 5pt; }}
.anexo {{ break-before: page; }}
</style></head><body><div class="wrap">

<h1>Anticipar el retraso: predicción de entregas tardías en un marketplace de e-commerce con validación temporal sin fuga de información</h1>
<div class="autores">{autores}</div>
<div class="afil">Equipo 3 · Universidad Anáhuac Querétaro · Diplomado Liderazgo en la Era de la IA: Estrategias de IA Generativa,
Business Intelligence y Ciencia de Datos · Proyecto final · Noviembre de 2026</div>

<div class="abstract"><h2>1. Resumen</h2>{resumen_txt}
<div class="kw"><b>Palabras clave:</b> clasificación desbalanceada, validación temporal, fuga de información, costo del error, logística de última milla, e-commerce.</div></div>

<h2>2. Introducción y definición del problema</h2>
<p><b>Empresa y contexto.</b> Olist es un marketplace brasileño que conecta a pequeños vendedores con compradores de todo el país. El
vendedor despacha, una transportista entrega y la plataforma promete al cliente una fecha de entrega. Cuando la promesa no se cumple, el
costo no es sólo logístico: la reseña pública del pedido se deteriora y, con ella, la reputación del vendedor y la conversión de futuros
compradores.</p>
<p><b>Decisión y destinatario.</b> El proyecto apoya una decisión que se toma todos los días: <i>qué pedidos recién comprados conviene atender de
forma proactiva</i>, ya sea avisando al cliente, escalando con el vendedor o la transportista, o ajustando la promesa. Quien la toma es el
responsable de operaciones logísticas y de experiencia del cliente, que tiene una capacidad limitada de atención por día.</p>
<p><b>Cuánto cuesta equivocarse.</b> La Tabla 1 resume, con los datos, el costo de cada tipo de error. Un falso negativo (un retraso que no se
anticipó) eleva en {f(rv['dp'] * 100, 1)} puntos porcentuales la probabilidad de una reseña de 1–2 estrellas, lo que equivale a unas
{n(cr['resenas_extra'])} reseñas negativas adicionales en los cuatro meses del periodo de prueba. Un falso positivo (una alerta sobre un pedido
que llegaría a tiempo) cuesta el contacto o la intervención. Como el modelo concentra los retrasos, revisando el 20 % de pedidos de mayor riesgo
se necesitan {f(equil[20]['Alertas por retraso anticipado'], 1)} alertas por cada retraso anticipado: la acción preventiva conviene si gestionar
a tiempo un retraso vale más que ese número de alertas.</p>
<div class="tcap"><b>Tabla 1.</b> Costo de cada tipo de error, estimado con los datos (notebook 07).</div>{tabla_costos()}
<p><b>Requerimiento de machine learning.</b> Dado un pedido en el instante <i>t</i> de la compra, estimar la probabilidad de que se entregue después
de la fecha prometida usando sólo información disponible en <i>t</i>. El objetivo operativo es ordenar los pedidos por riesgo para capturar la mayor
fracción de retrasos dentro de una capacidad fija de revisión (<i>k</i> % de los pedidos). El modelo se considera útil si supera al azar y a la
mejor regla de una sola variable elegida con datos de entrenamiento.</p>
<p><b>Contribuciones.</b> (i) Un conjunto de variables construido estrictamente con información del momento de la compra y verificado contra fugas;
(ii) un esquema de validación temporal con purga que refleja cuándo se conoce realmente la etiqueta de un pedido; (iii) una comparación controlada
de cuatro familias de modelos, cada una con su propia selección de variables y ajuste de hiperparámetros, y contra reglas simples; y (iv) una
traducción de los resultados a costos y decisiones operativas.</p>

<h2>3. Estado del arte</h2>
<p>La predicción de tiempos de entrega se ha estudiado sobre todo como un problema de regresión con información rica en tiempo real. Salari et al.
(2022) estiman con modelos de árboles la distribución del tiempo de entrega de los pedidos de JD.com y la usan para fijar la promesa, con un aumento
de 6,1 % en ventas. En movilidad, Derrow-Pinion et al. (2021) predicen el tiempo de llegada en Google Maps con redes neuronales de grafos sobre la red
vial, y Hu et al. (2022) proponen en Uber un modelo que corrige el residuo de un motor de rutas. Estos trabajos muestran el valor de una buena
promesa, pero dependen de tráfico, rutas o escaneos de la transportista que no existen en el momento de la compra. Nuestro problema es distinto: la
promesa ya está dada y queremos anticipar, sólo con la información de la compra, si se incumplirá.</p>
<p>Sobre la elección de modelos, Grinsztajn et al. (2022) muestran en 45 conjuntos tabulares que los modelos de árboles siguen superando a las redes
profundas en tablas de tamaño mediano, y Shwartz-Ziv y Armon (2022) llegan a una conclusión parecida al comparar arquitecturas recientes contra
XGBoost. Por eso comparamos regresión logística, Random Forest y XGBoost contra una red neuronal, sin presuponer que la red gane. Sobre la evaluación,
Kaufman et al. (2012) formalizan la fuga de información, Bergmeir y Benítez (2012) y López de Prado (2018) discuten la validación cruzada con datos
temporales y la purga entre periodos, Saito y Rehmsmeier (2015) recomiendan la curva precisión-recall para clases desbalanceadas y Elkan (2001)
relaciona el umbral de decisión con el costo de cada error. Nuestro aporte combina estas piezas en un caso de marketplace (Tabla 2).</p>
<div class="tcap"><b>Tabla 2.</b> Trabajos relacionados.</div>{tabla_arte()}

<h2>4. Datos</h2>
<h3>4.1 Fuente, permiso de uso y periodo</h3>
<p>Usamos el <i>Brazilian E-Commerce Public Dataset by Olist</i> (Olist, 2018), publicado en Kaggle con licencia Creative Commons
Atribución-NoComercial-CompartirIgual 4.0 (CC BY-NC-SA 4.0), que permite su uso académico citando la fuente. Los datos son reales y están
anonimizados: no contienen nombres ni datos que identifiquen a personas. Comprenden {n(limp['pedidos_originales'])} pedidos de septiembre de 2016 a
octubre de 2018 en nueve archivos relacionales: pedidos, ítems, pagos, productos, vendedores, clientes, geolocalización por código postal, reseñas y
traducción de categorías. Las tablas se unieron y prepararon con pandas en los notebooks 01 a 04. El conjunto de modelado tiene
{n(eda['pedidos'])} pedidos (enero 2017 – agosto 2018) descritos por {n_base} variables y la variable objetivo; el diccionario de datos está en el
Anexo A.</p>
<h3>4.2 Limpieza y variable objetivo</h3>
<p>Las reglas de limpieza fueron: eliminar pedidos cancelados o no disponibles ({n(limp['cancelados_o_no_disponibles'])}); excluir, por decisión
del equipo, pedidos sin fecha de entrega ({n(limp['sin_fecha_de_entrega'])}); y excluir pedidos cuya fecha estimada es posterior a la última
entrega registrada, para los cuales aún no se podría conocer el resultado ({n(limp['censurados_fecha_estimada_posterior'])}). Los pedidos de 2016
son escasos y presentan un hueco entre noviembre y diciembre; se usaron sólo como historial para las variables. No se eliminaron valores atípicos,
porque los pedidos extremos (muy pesados o lejanos) son precisamente los de mayor riesgo.</p>
<p>Un pedido es tardío (<code>delivery_late = 1</code>) si se entregó en un día posterior a la fecha estimada. La fecha estimada siempre viene a las
00:00, por lo que la comparación se hace por día: un pedido entregado el mismo día prometido cuenta como a tiempo. Definimos además el instante en
que la etiqueta se vuelve conocida, <code>label_conocida_en = min(entrega, estimada + 1 día)</code>: un pedido puntual se sabe al entregarse y uno
tardío en cuanto vence su promesa. Esta columna es la base de las variables históricas y de la purga.</p>
<h3>4.3 Análisis exploratorio y estadística descriptiva</h3>
<p>La tasa de retraso es de {p(tasas['global']['p'], 2)} (IC 95 % de Wilson [Wilson, 1927]: {f(tasas['global']['li'] * 100, 2)}–{f(tasas['global']['ls'] * 100, 2)} %),
lo que hace al problema desbalanceado. Los pedidos tardíos promedian {f(r['score_tardio'], 2)} estrellas frente a {f(r['score_a_tiempo'], 2)} de los puntuales
(Figura 1) y los retrasos llegan en oleadas (Figura 2): la tasa mensual pasó de menos de 3 % a {f(max(eda['mensual']['tasa']) * 100, 0)} % en marzo de 2018.
El pedido típico llega {f(eda['dias_antes_mediana'], 0)} días antes de lo prometido y, cuando un pedido se retrasa, el retraso mediano es de
{f(eda['dias_retraso_mediana_tardios'], 0)} días.</p>
<figure class="m"><img src="{fig}eda_reviews.png"><figcaption><b>Figura 1.</b> Distribución de calificaciones según puntualidad del pedido. Las reseñas
se usan sólo para dimensionar el problema, nunca como variable del modelo.</figcaption></figure>
<figure><img src="{fig}eda_tasa_mensual.png"><figcaption><b>Figura 2.</b> Volumen de pedidos y tasa de retraso por mes de compra. La línea punteada marca el
inicio del periodo de prueba, que tiene una tasa de retraso menor ({Te['% tardíos']} %) que el de entrenamiento ({T['% tardíos']} %).</figcaption></figure>
<p>La Tabla 3 describe las variables clave en el periodo de entrenamiento. Como tamaño del efecto reportamos el AUC univariado: la probabilidad de
que un pedido tardío tenga un valor mayor que uno puntual (0,5 = no separa). Ninguna variable aislada pasa de 0,67; las que más separan son la carga de
pedidos de los últimos siete días, el historial de la ruta y la tasa de retraso reciente hacia el estado del cliente. Los días prometidos casi no
separan por sí solos (0,49), porque la promesa ya incorpora un margen: su efecto aparece al combinarlos con otras variables.</p>
<div class="tcap"><b>Tabla 3.</b> Estadística descriptiva por clase, periodo de entrenamiento ({n(comp['n_train_descriptiva'])} pedidos).</div>{tabla_descriptiva()}
<p>La Tabla 4 respalda con intervalos de confianza y pruebas de hipótesis los hallazgos que usamos en el resto del trabajo: el costo del retraso en
las reseñas, el cambio de la tasa de retraso entre periodos (deriva), que motiva la validación temporal, y el efecto de la saturación reciente, que
motiva las variables de carga. Por cuartil de la tasa reciente hacia el estado, la tasa de retraso sube de forma monótona
({", ".join(p(c['% tardíos'], 1) for c in carga)}); con la tasa global del sistema la relación no es monótona, porque el cuartil más alto coincide con la
salida de las oleadas. La recompra a 180 días no muestra una diferencia significativa.</p>
<div class="tcap"><b>Tabla 4.</b> Intervalos de confianza al 95 % y pruebas de hipótesis (notebook 07).</div>{tabla_inferencia()}

<h2>5. Metodología</h2>
<h3>5.1 División en entrenamiento y prueba y validación cruzada</h3>
<p>El periodo de prueba son los pedidos comprados a partir del {split['fecha_corte']} ({n(Te['Filas'])} pedidos, {Te['% tardíos']} % tardíos); no participó
en ninguna decisión y se evaluó una sola vez. El entrenamiento ({n(T['Filas'])} pedidos, {T['% tardíos']} % tardíos) se dividió en seis bloques
cronológicos para formar cinco folds de ventana creciente: el fold <i>k</i> entrena con los bloques 1…<i>k</i> y valida con el bloque <i>k</i>+1 (Tabla 5).
A diferencia de <code>TimeSeriesSplit</code>, aplicamos <b>purga</b> (López de Prado, 2018): un pedido sólo entrena si su etiqueta ya se conocía al
inicio del periodo de validación. Sin ella, los últimos pedidos del entrenamiento tendrían etiquetas que en producción todavía no existirían. Entre
entrenamiento y prueba se purgaron {n(split['purgados'])} pedidos. La imputación (mediana) y el escalado se ajustan dentro de cada fold mediante
<code>Pipeline</code>.</p>
<div class="tcap"><b>Tabla 5.</b> Folds temporales con purga.</div>{tabla_folds()}
<figure><img src="{fig}split_temporal.png"><figcaption><b>Figura 3.</b> Esquema de validación: cinco folds de ventana creciente y periodo de prueba
aislado. La purga retira del entrenamiento los pedidos cuyo resultado aún no se conocía al inicio de cada validación (huecos apenas visibles a esta escala).</figcaption></figure>
<h3>5.2 Variables</h3>
<p>Cada pedido se describe con {n_base} variables agrupadas en ocho familias: promesa de entrega (días prometidos y plazo del vendedor para despachar),
tamaño del pedido, características físicas del producto, forma de pago, geografía (estados, distancia de Haversine entre los centroides de los códigos
postales), calendario (día de la semana y franja horaria), historial y carga reciente. El mes de compra se descartó porque, con menos de dos años de
datos, identifica episodios concretos que el modelo memorizaría en lugar de estacionalidad.</p>
<p>Las variables de historial (tasa de retraso del vendedor y de la ruta estado→estado) y de carga reciente (tasa de retraso de las entregas conocidas
en los 30 días previos, global y hacia el estado del cliente, y número de pedidos de los últimos siete días) son <i>point-in-time</i>: para un pedido
comprado en <i>t</i> sólo cuentan etiquetas con <code>label_conocida_en &lt; t</code> (Kaufman et al., 2012). Las tasas se suavizan hacia un
<i>prior</i> con m = {config.M_SUAVIZADO} observaciones; el prior es la tasa global conocida hasta <i>t</i>, no la del periodo de entrenamiento completo, que
filtraría información futura hacia los primeros folds. Verificamos la ausencia de fuga recalculando por fuerza bruta el historial de 300 pedidos
aleatorios. Las variables categóricas se codificaron con <i>one-hot</i> (las {config.MAX_CATEGORIAS} categorías más frecuentes en entrenamiento), para un
total de {split['n_variables']} columnas.</p>
<h3>5.3 Modelos clásicos, selección de variables e hiperparámetros</h3>
<p>Se compararon tres modelos clásicos implementados con scikit-learn (Pedregosa et al., 2011): regresión logística con pesos balanceados (línea
base), Random Forest (Breiman, 2001) y XGBoost (Chen y Guestrin, 2016). Para cada uno se siguió la misma secuencia: (1) eliminación recursiva de
variables con validación cruzada (RFECV; Guyon et al., 2002), retirando el 10 % de variables menos importantes por ronda según los coeficientes o la
importancia propia del modelo; y (2) búsqueda aleatoria de hiperparámetros (Bergstra y Bengio, 2012) sobre las variables elegidas. Ambos pasos usan
los mismos cinco folds y la métrica PR-AUC.</p>
<h3>5.4 Red neuronal</h3>
<p>La red es un perceptrón multicapa en Keras (Chollet et al., 2015): capas densas con activación ReLU, regularización L2 y <i>dropout</i>
(Srivastava et al., 2014), salida sigmoide, pérdida de entropía cruzada ponderada por clase y optimizador Adam. Como la red no ofrece una medida de
importancia directa, su selección de variables comparó conjuntos candidatos (todas, el RFECV de cada modelo clásico, su unión y su intersección),
seguida de una búsqueda aleatoria sobre arquitectura, <i>dropout</i>, L2, tasa de aprendizaje y tamaño de lote. Para evitar el sobreajuste se usó
<b>detención temprana</b> (Prechelt, 1998): en cada fold, la pérdida se vigila sobre el 15 % más reciente del entrenamiento (nunca sobre el fold de
validación), con paciencia de 8 épocas, un máximo de 100 y recuperación de los mejores pesos. La configuración elegida tiene capas
{fmt_param(params[NN]['capas'])}, <i>dropout</i> {fmt_param(params[NN]['dropout'])}, L2 {fmt_param(params[NN]['l2'])}, tasa de aprendizaje
{fmt_param(params[NN]['lr'])} y lotes de {params[NN]['batch']}, con {resumen[NN]['n_variables']} variables (conjunto {resumen[NN]['conjunto_variables']}).
En el ajuste final la mejor época fue la {resumen[NN]['mejor_epoca']} y el entrenamiento se detuvo ocho épocas después (Figura 4).</p>
<figure><img src="{fig}curvas_red_neuronal.png"><figcaption><b>Figura 4.</b> Curvas de aprendizaje del ajuste final de la red, medidas sin
<i>dropout</i>. La pérdida del tramo de detención temprana se estabiliza y el entrenamiento se detiene al dejar de mejorar; se conservan los pesos de
la mejor época. El tramo de detención (los meses más recientes) es más fácil de ordenar que el promedio del entrenamiento, por eso su ROC-AUC es mayor.</figcaption></figure>
<h3>5.5 Métricas, umbral y regla de selección</h3>
<p>Con {p(eda['tasa_tardios'])} de positivos, la exactitud no es informativa: predecir siempre «a tiempo» acierta más del 93 %. Las métricas se eligieron
por el costo de cada error. Como un falso negativo cuesta mucho más que una alerta (Tabla 1) y la operación tiene capacidad limitada, lo que importa es
<b>ordenar</b> bien los pedidos: la métrica principal es el área bajo la curva precisión-recall (PR-AUC, <i>average precision</i>; Saito y Rehmsmeier,
2015), cuyo valor de azar es la tasa de positivos, y la complementamos con la captura de retrasos en el <i>k</i> % de pedidos de mayor riesgo. Reportamos
también ROC-AUC, matriz de confusión, precisión, sensibilidad (<i>recall</i>) y F1 de la clase tardía. El sobreajuste se diagnostica con la brecha de
ROC-AUC entre entrenamiento y validación, porque el PR-AUC depende de la tasa de positivos y ésta cambia entre periodos. El umbral de cada modelo es el
que maximiza el F1 sobre sus predicciones fuera de fold; con probabilidades calibradas, el umbral óptimo por costo sería
p* = C<sub>FP</sub> / (C<sub>FP</sub> + C<sub>FN</sub>) (Elkan, 2001). La regla de selección se fijó antes de ver resultados: mayor PR-AUC promedio en
validación y, en caso de empate (diferencia menor que {f(config.TOLERANCIA_EMPATE, 3)}), el modelo más simple (logística, Random Forest, XGBoost, red
neuronal).</p>

<h2>6. Resultados</h2>
<h3>6.1 Comparación en validación temporal</h3>
<p>La Tabla 6 resume el desempeño en los cinco folds. {frase_empate} {frase_nn} (PR-AUC {f(cv[NN]['PR-AUC val (prom)'])} frente a
{f(cv[mejor]['PR-AUC val (prom)'])}). La desviación entre folds es grande para todos los modelos y sigue a la tasa de retraso de cada periodo
(Figura 5): los folds con oleadas de retraso son más fáciles de ordenar y el primero, con menos de 3 % de positivos y sólo cinco meses de historia, es
el más difícil. La brecha de ROC-AUC entre entrenamiento y validación de la red ({f(cv[NN]['Brecha ROC train-val'])}) es similar a la de la
regresión logística ({f(cv['Regresión Logística']['Brecha ROC train-val'])}) y menos de la mitad que la de Random Forest
({f(cv['Random Forest']['Brecha ROC train-val'])}) y XGBoost ({f(cv['XGBoost']['Brecha ROC train-val'])}): la detención temprana controló el sobreajuste.</p>
<div class="tcap"><b>Tabla 6.</b> Validación cruzada temporal (promedio de 5 folds). «× azar» = PR-AUC / tasa de tardíos fuera de fold ({p(res['azar_cv'], 2)}).
Precisión y recall con el umbral de cada modelo.</div>{tabla_cv()}
<figure><img src="{fig}comparacion_cv.png"><figcaption><b>Figura 5.</b> Izquierda: PR-AUC de validación (promedio ± desviación). Centro: ROC-AUC de
entrenamiento (gris) y validación (color); la brecha indica sobreajuste. Derecha: curvas precisión-recall de las predicciones fuera de fold.</figcaption></figure>
<div class="tcap"><b>Tabla 7.</b> Variables e hiperparámetros elegidos.</div>{tabla_params()}

<h3>6.2 Prueba final</h3>
<p>El periodo de prueba se abrió una sola vez para el mejor modelo clásico y la red neuronal (Tabla 8). {frase_test} Los valores absolutos son menores
que en validación porque el periodo de prueba tiene menos retrasos ({p(res['azar_test'])}); la mejora relativa frente al azar es de
{f(te[mejor]['PR-AUC / azar'], 1)} veces para {mejor} y {f(te[NN]['PR-AUC / azar'], 1)} para la red. El umbral fijo de probabilidad elegido en validación
no se transfirió bien: en prueba marca el {f(marcados * 100, 0)} % de los pedidos, con recall alto ({p(te[mejor]['Recall'], 0)}) pero precisión baja
({p(te[mejor]['Precision'], 0)}) (Figura 6). Esto ocurre porque la tasa de retraso bajó y porque el umbral se eligió sobre predicciones de cinco modelos
de fold con escalas de probabilidad distintas. La PR-AUC y la operación por capacidad son robustas a esa deriva.</p>
<div class="tcap"><b>Tabla 8.</b> Desempeño en el periodo de prueba ({n(res['n_test'])} pedidos, {n(res['tardios_test'])} tardíos). El umbral se eligió en validación.</div>{tabla_test()}
<figure><img src="{fig}test_final.png"><figcaption><b>Figura 6.</b> Matrices de confusión y curvas precisión-recall en el periodo de prueba.</figcaption></figure>

<h3>6.3 Operación por capacidad y punto de equilibrio</h3>
<p>Operar por capacidad significa revisar cada día el <i>k</i> % de pedidos con mayor riesgo, en lugar de usar un umbral de probabilidad. Revisando el
10 % se anticipa el {f(cap[10]['% tardíos capturados'], 0)} % de los retrasos con unas {f(equil[10]['Alertas'] / dias_test, 0)} alertas diarias; revisando
el 20 %, el {f(cap[20]['% tardíos capturados'], 0)} % con unas {f(equil[20]['Alertas'] / dias_test, 0)} (Tabla 9). La última columna da el punto de
equilibrio: cuántas alertas cuesta cada retraso anticipado y, por tanto, cuántas veces debe valer gestionar un retraso frente al costo de una alerta
para que la acción convenga.</p>
<div class="tcap"><b>Tabla 9.</b> Captura de retrasos y punto de equilibrio al revisar el <i>k</i> % de pedidos con mayor riesgo (prueba, {dias_test} días).</div>{tabla_capacidad()}

<h3>6.4 Comparación contra la solución más simple</h3>
<p>Para saber si hace falta un modelo, lo comparamos con reglas de una sola variable que un analista podría usar sin machine learning (Tabla 10,
Figura 7). Para que la comparación sea justa, la regla se eligió igual que el modelo: con el periodo de entrenamiento. La regla elegida
(◆ {elegida['Método'].replace('Regla: ', '')}) obtiene en prueba una PR-AUC de {f(elegida['PR-AUC'])}, apenas {f(elegida['× azar'], 1)} veces el azar,
frente a {f(te[mejor]['PR-AUC'])} de {mejor_corto}: varias señales que funcionaban en entrenamiento se desploman en el periodo de prueba, mientras que
los modelos multivariables conservan su ventaja. Por honestidad reportamos también un resultado <i>a posteriori</i>: la regla de
{posthoc['Método'].replace('Regla: ', '').lower()} (◇) alcanza en prueba una PR-AUC de {f(posthoc['PR-AUC'])}, mayor que la de los modelos, aunque con
menor ROC-AUC ({f(posthoc['ROC-AUC'])} frente a {f(te[mejor]['ROC-AUC'])}) y una captura similar en el 20 % de mayor riesgo. En entrenamiento esa regla
tenía una PR-AUC de {f(posthoc['PR-AUC train'])} y no se habría elegido: la importancia de la promesa de entrega cambió en el periodo de prueba, lo que
refuerza la necesidad de reentrenar y monitorear el modelo.</p>
<div class="tcap"><b>Tabla 10.</b> Modelos frente a reglas de una sola variable en el periodo de prueba. ◆ regla elegida con entrenamiento;
◇ mejor regla sólo a posteriori.</div>{tabla_baselines()}
<figure><img src="{fig}baselines_test.png"><figcaption><b>Figura 7.</b> PR-AUC en el periodo de prueba de los modelos y de las reglas simples. La línea
punteada es el azar.</figcaption></figure>

<h3>6.5 Variables relevantes</h3>
<p>Las variables con mayor peso en {mejor} son: {lista_imp}. El patrón es coherente con el análisis exploratorio: la promesa de entrega, la saturación
reciente del sistema y el historial de la ruta y del vendedor explican la mayor parte del riesgo.</p>
<figure class="m"><img src="{fig}importancia_final.png"><figcaption><b>Figura 8.</b> Importancia relativa de las variables del mejor modelo clásico.</figcaption></figure>

<h2>7. Retos, límites y conclusión</h2>
<h3>7.1 Qué no se puede concluir con estos datos</h3>
<ul>
<li><b>Causalidad en las reseñas.</b> La caída de calificación de los pedidos tardíos es una asociación; parte podría deberse a factores que acompañan
al retraso (por ejemplo, vendedores con peor servicio).</li>
<li><b>Pérdida de clientes.</b> No encontramos evidencia de que el retraso reduzca la recompra a 180 días (p = {pv(rec['p'])}); con una recompra general
cercana al 3 %, estos datos no tienen potencia para medir ese efecto.</li>
<li><b>Superioridad de la red.</b> La ventaja de la red en validación no se sostuvo en prueba: no podemos afirmar que sea mejor que la regresión logística.</li>
<li><b>Efecto de las alertas.</b> El modelo anticipa retrasos, pero no sabemos cuántas reseñas negativas evitaría una alerta; eso sólo se mide con un
experimento controlado.</li>
<li><b>Generalización.</b> Los resultados corresponden a Olist en Brasil entre 2017 y 2018; no se pueden extrapolar a otros mercados o periodos sin
validarlos.</li>
</ul>
<h3>7.2 Riesgos de usar el modelo</h3>
<ul>
<li><b>Deriva temporal.</b> La tasa de retraso cambió de forma significativa entre periodos (χ² = {f(comp['drift']['chi2'], 0)}, p {pv(comp['drift']['p'])})
y varía de 1 % a 19 % entre meses (Gama et al., 2014); la importancia de las variables también cambió (sección 6.4).</li>
<li><b>Umbral frágil.</b> Un umbral fijo de probabilidad marcó el {f(marcados * 100, 0)} % de los pedidos en prueba; recomendamos fijar la capacidad de
revisión o recalibrar periódicamente.</li>
<li><b>Retroalimentación.</b> Si las alertas evitan retrasos, las etiquetas futuras dejarán de reflejar el riesgo original; hay que registrar qué pedidos
se intervinieron para reentrenar sin sesgo.</li>
<li><b>Trato desigual por región.</b> El riesgo se concentra en algunos estados (Alagoas {f(estados['AL'] * 100, 0)} %, Maranhão
{f(estados['MA'] * 100, 0)} %, Ceará {f(estados['CE'] * 100, 0)} % frente a São Paulo {f(estados['SP'] * 100, 1)} % en entrenamiento). Las alertas
deben mejorar el servicio en esas regiones, no restringir la venta ni encarecer el envío, y conviene monitorear el desempeño por estado.</li>
<li><b>Optimismo del CV.</b> Los mismos folds eligieron variables, hiperparámetros y umbral, por lo que las métricas de validación son optimistas; por eso
reportamos un periodo de prueba aislado. En iteraciones anteriores del proyecto se consultó el periodo de prueba; la versión final fijó todas las
decisiones con validación cruzada antes de evaluarlo.</li>
<li><b>Variables ausentes.</b> No hay datos de transportista, clima, huelgas o festivos, que explican oleadas como la de marzo de 2018.</li>
</ul>
<h3>7.3 Recomendación</h3>
<p>Recomendamos lanzar un <b>piloto de alertas proactivas por capacidad</b> con {mejor_corto}: revisar cada día el 10–20 % de pedidos con mayor
riesgo (unas {f(equil[10]['Alertas'] / dias_test, 0)}–{f(equil[20]['Alertas'] / dias_test, 0)} alertas diarias con el volumen del periodo de prueba), lo
que anticipa el {f(cap[10]['% tardíos capturados'], 0)}–{f(cap[20]['% tardíos capturados'], 0)} % de los retrasos. Para acciones baratas, como avisar al
cliente, conviene el extremo alto de ese rango; para intervenciones costosas, el bajo. La red neuronal debe quedar como modelo retador
(<i>champion–challenger</i>), con reentrenamiento mensual con el mismo esquema temporal y monitoreo de la PR-AUC frente al azar. El piloto debe medirse con
un experimento A/B cuyo indicador sea la proporción de reseñas de 1–2★ en pedidos tardíos.</p>
<h3>7.4 Siguientes pasos</h3>
<p>(i) Incorporar fuentes exógenas (transportista, clima, calendario de festivos); (ii) calibrar las probabilidades para fijar umbrales por
costo; (iii) modelar también los días de retraso; y (iv) validar el impacto con el experimento del piloto.</p>
<h3>7.5 Conclusión</h3>
<p>Con información disponible al momento de la compra es posible anticipar una parte relevante de las entregas tardías de Olist: los modelos ordenan
el riesgo {f(te[mejor]['PR-AUC / azar'], 1)} veces mejor que el azar en meses que nunca vieron y superan a la regla simple que se habría elegido sin
machine learning. El aporte principal no es la complejidad del modelo, sino el diseño de evaluación: variables <i>point-in-time</i>, validación temporal
con purga, una regla de selección fijada de antemano y una lectura de los resultados en términos de costo y capacidad.</p>
<p><b>Reproducibilidad.</b> El código está en https://github.com/fabiolajrzflo/proyecto_final_equipo_3, organizado en notebooks secuenciales
(<code>01_limpieza</code> a <code>07_estadistica_y_costo_del_error</code>) y módulos en <code>src/</code>; todas las cifras de este documento se
generan automáticamente a partir de sus salidas.</p>

<h2>8. Referencias</h2>
<div class="ref">
<p>Bergmeir, C., y Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. <i>Information Sciences, 191</i>, 192–213. https://doi.org/10.1016/j.ins.2011.12.028</p>
<p>Bergstra, J., y Bengio, Y. (2012). Random search for hyper-parameter optimization. <i>Journal of Machine Learning Research, 13</i>, 281–305. https://jmlr.org/papers/v13/bergstra12a.html</p>
<p>Breiman, L. (2001). Random forests. <i>Machine Learning, 45</i>(1), 5–32. https://doi.org/10.1023/A:1010933404324</p>
<p>Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i> (pp. 785–794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785</p>
<p>Chollet, F., et al. (2015). <i>Keras</i> [Software]. https://keras.io</p>
<p>Derrow-Pinion, A., She, J., Wong, D., Lange, O., Hester, T., Perez, L., Nunkesser, M., Lee, S., Guo, X., Wiltshire, B., Battaglia, P. W., Gupta, V., Li, A., Xu, Z., Sanchez-Gonzalez, A., Li, Y., y Veličković, P. (2021). ETA prediction with graph neural networks in Google Maps. En <i>Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management</i> (pp. 3767–3776). Association for Computing Machinery. https://doi.org/10.1145/3459637.3481916</p>
<p>Elkan, C. (2001). The foundations of cost-sensitive learning. En <i>Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence</i> (pp. 973–977). Morgan Kaufmann.</p>
<p>Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., y Bouchachia, A. (2014). A survey on concept drift adaptation. <i>ACM Computing Surveys, 46</i>(4), Artículo 44. https://doi.org/10.1145/2523813</p>
<p>Grinsztajn, L., Oyallon, E., y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? En <i>Advances in Neural Information Processing Systems 35: Datasets and Benchmarks Track</i>. https://doi.org/10.48550/arXiv.2207.08815</p>
<p>Guyon, I., Weston, J., Barnhill, S., y Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. <i>Machine Learning, 46</i>(1–3), 389–422. https://doi.org/10.1023/A:1012487302797</p>
<p>Hu, X., Binaykiya, T., Frank, E., y Cirit, O. (2022). <i>DeeprETA: An ETA post-processing system at scale</i> (arXiv:2206.02127). arXiv. https://doi.org/10.48550/arXiv.2206.02127</p>
<p>Kaufman, S., Rosset, S., Perlich, C., y Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. <i>ACM Transactions on Knowledge Discovery from Data, 6</i>(4), 1–21. https://doi.org/10.1145/2382577.2382579</p>
<p>López de Prado, M. (2018). <i>Advances in financial machine learning</i>. Wiley.</p>
<p>Olist. (2018). <i>Brazilian e-commerce public dataset by Olist</i> [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce</p>
<p>Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html</p>
<p>Prechelt, L. (1998). Early stopping — But when? En G. B. Orr y K.-R. Müller (Eds.), <i>Neural networks: Tricks of the trade</i> (pp. 55–69). Springer. https://doi.org/10.1007/3-540-49430-8_3</p>
<p>Saito, T., y Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. <i>PLOS ONE, 10</i>(3), Artículo e0118432. https://doi.org/10.1371/journal.pone.0118432</p>
<p>Salari, N., Liu, S., y Shen, Z.-J. M. (2022). Real-time delivery time forecasting and promising in online retailing: When will your package arrive? <i>Manufacturing &amp; Service Operations Management, 24</i>(3), 1421–1436. https://doi.org/10.1287/msom.2022.1081</p>
<p>Shwartz-Ziv, R., y Armon, A. (2022). Tabular data: Deep learning is not all you need. <i>Information Fusion, 81</i>, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011</p>
<p>Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., y Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. <i>Journal of Machine Learning Research, 15</i>, 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html</p>
<p>Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. <i>Journal of the American Statistical Association, 22</i>(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953</p>
</div>

<div class="anexo">
<h2>Anexo A. Diccionario de datos</h2>
<p>Variables del conjunto de modelado (<code>datos/processed/dataset_pedidos.parquet</code>), un registro por pedido. Los identificadores
(<code>order_id</code>, <code>customer_id</code>, <code>main_seller_id</code>, <code>ruta</code>, código postal) y las fechas no entran al modelo.
Las variables categóricas se codifican con <i>one-hot</i> antes de entrenar.</p>
<div class="tcap"><b>Tabla A1.</b> Diccionario de datos.</div>{tabla_diccionario()}
</div>
</div></body></html>"""

carpeta = RAIZ / "reportes" / "articulo"
carpeta.mkdir(parents=True, exist_ok=True)
salida = carpeta / "articulo.html"
salida.write_text(html)
print(f"HTML: {salida}  (resumen: {palabras} palabras)")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
pdf = carpeta / "reporte_tecnico.pdf"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                f"--print-to-pdf={pdf}", salida.as_uri()], check=True, capture_output=True)
print("PDF:", pdf)
