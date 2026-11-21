# Guía rápida para estudiar · Proyecto Olist

> La versión corta de la [guía detallada](guia_detallada_notebooks.md). Léela hasta que puedas contar cada notebook sin mirar.

---

## 1. La historia en 30 segundos

> Olist es un marketplace de Brasil. **6.8 %** de los pedidos llega tarde y eso **hunde la reseña** (2.3★ vs. 4.3★).
> Construimos un modelo que, **el día de la compra**, estima la probabilidad de que el pedido llegue tarde.
> Lo hicimos **sin hacer trampa**: sólo con información que existía ese día, dividiendo los datos por tiempo y abriendo el examen final (test) **una sola vez**.
> Resultado: el modelo ordena los pedidos por riesgo **≈ 2.2 veces mejor que el azar**. Revisando el **20 %** de pedidos más riesgosos se anticipa el **39 %** de los retrasos.

---

## 2. Mapa del proyecto

| # | Notebook | En una frase | Entra → Sale |
|---|---|---|---|
| 01 | Limpieza | Arreglar datos y definir "tarde" | CSV → tablas limpias |
| 02 | Exploración | Encontrar las historias en los datos | tablas limpias → gráficas |
| 03 | Variables | Una fila por pedido con lo que se sabe al comprar | tablas limpias → dataset |
| 04 | Preparación | Separar train/test por tiempo y crear 5 folds | dataset → X_train, X_test, folds |
| 05a–c | Modelos clásicos | Elegir variables y perillas de cada modelo | X_train → modelo guardado |
| 05d | Red neuronal | Lo mismo, adaptado a una red | X_train → red guardada |
| 06 | Final | Comparar con CV, abrir el test una vez | modelos → resultados |

**Truco para recordarlo:** *Limpio → Miro → Construyo → Separo → Entreno → Decido.*

---

## 3. Cada notebook en 5 líneas

### 01 · Limpieza
- Lee 8 tablas (`cargar_tablas`).
- Arregla códigos postales (5 dígitos), estados válidos, coordenadas dentro de Brasil y peso 0 → faltante.
- Quita pedidos cancelados, sin fecha de entrega y "censurados" (del final, cuyo resultado todavía no se sabría).
- **Label:** `delivery_late = 1` si llegó un día después de lo prometido.
- **`label_conocida_en`:** cuándo se supo el resultado = la entrega, o la fecha prometida + 1 día si llegó tarde.

**Frase para el jurado:** *"De 99,441 pedidos quedaron 96,468 limpios; documentamos cada fila eliminada."*

### 02 · Exploración
- Reseñas: tardío 2.3★ vs. a tiempo 4.3★ → **el caso de negocio**.
- Tasa mensual: **oleadas** (Black Friday 2017 12 %, marzo 2018 19 %).
- El pedido típico llega 12 días **antes** de lo prometido (la fecha trae "colchón").
- No se borran outliers (los extremos son los que más se retrasan).
- Todo análisis contra el label se hace **sólo con train**.

**Frase:** *"El retraso es poco frecuente pero muy caro en satisfacción, y llega en oleadas."*

### 03 · Variables
- `construir_dataset`: 36 variables en 8 grupos (promesa de entrega, pedido, producto, pago, geografía, calendario, historial, carga reciente).
- **Point-in-time:** el historial del vendedor y la ruta sólo cuenta resultados ya conocidos antes de la compra (`merge_asof`).
- **Suavizado (m = 20):** un vendedor con pocos pedidos se parece al promedio general.
- **Prueba de fuerza bruta** con 300 pedidos: 0 errores → no hay trampa.
- El mes de compra **no** se usa (memorizaría episodios).

**Frase:** *"Cada variable usa sólo lo que la empresa sabía el día de la compra, y lo verificamos."*

### 04 · Preparación
- **Test:** compras desde el 1 de mayo de 2018 (25,350 pedidos, 4.4 % tardíos).
- **Train:** antes del corte (68,287 pedidos, 7.6 % tardíos).
- **Purga:** 2,564 pedidos fuera, porque su resultado se supo después del corte.
- **5 folds de ventana creciente:** entrenar con el pasado, validar con el bloque siguiente (y purgar también).
- **One-hot:** categorías → columnas 0/1 (aprendido sólo con train) → 105 columnas.

**Frase:** *"Es el diagrama del profesor: folds para elegir, test aparte para el examen final, y purga para no ver el futuro."*

### 05a–c · Modelos clásicos (misma plantilla)
1. **RFECV:** quita el 10 % de variables menos útiles por ronda y se queda con el mejor número.
2. **RandomizedSearchCV:** prueba combinaciones al azar de hiperparámetros.
3. **Evaluación en 5 folds:** PR-AUC para elegir; ROC-AUC train vs. validación para ver sobreajuste; umbral = F1 máximo.
4. **Re-entrena con todo el train y guarda** (`modelos/<modelo>/`).

| Modelo | Variables | PR-AUC val | Brecha ROC (sobreajuste) |
|---|---|---|---|
| Logística | 65 | 0.153 | 0.08 (baja) |
| Random Forest | 105 | 0.158 | 0.21 (alta) |
| XGBoost | 35 | 0.157 | 0.20 (alta) |

**Frase:** *"Cada modelo eligió sus propias variables y perillas con los mismos folds. Los tres empataron."*

### 05d · Red neuronal
- MLP (capas densas), porque los datos son una tabla.
- Selección de variables: comparar 6 listas candidatas → ganó la de XGBoost (35 variables).
- Búsqueda de 6 configuraciones → `(64, 32, 16)`, dropout 0.1, L2 0.01.
- Control del sobreajuste: **early stopping** (con el 15 % más reciente del train), **Dropout** y **L2**.
- PR-AUC val **0.190**: la mejor en validación.

**Frase:** *"La red siguió el mismo flujo, adaptado: no tiene coeficientes, así que comparamos listas de variables."*

### 06 · Selección y test
- **Regla fijada de antemano:** mayor PR-AUC; si hay empate (< 0.005), gana el más simple.
- Clásicos empatados → **Logística**. La red gana el CV.
- **Test (una vez):** Logística 0.098 vs. red 0.095 → **equivalentes**, ≈ 2.2 veces el azar.
- El umbral fijo de validación marca demasiados pedidos en test (deriva) → conviene **operar por capacidad** (top *k* %).
- Recomendación: **Logística** (simple, explicable) con la red como **retadora**.

**Frase:** *"La red ganó en validación pero no en el futuro; con desempeño igual, preferimos el modelo más simple."*

---

## 4. Los 12 conceptos que debes dominar

| Concepto | Explicación de secundaria |
|---|---|
| **Fuga de información** | Hacer trampa viendo datos del futuro. |
| **Point-in-time** | Sólo usar lo que se sabía en ese momento. |
| **Train / Test** | Estudiar con exámenes pasados / examen final que se abre una vez. |
| **Fold** | Examen de práctica: entreno con lo viejo, me pruebo con lo siguiente. |
| **Purga** | No estudiar con ejercicios cuya respuesta aún no te daban. |
| **One-hot** | Convertir "SP", "RJ"… en columnas de sí/no. |
| **Imputar** | Rellenar huecos (con la mediana del train). |
| **Pipeline** | Línea de producción: rellenar → escalar → modelo. |
| **RFECV** | Quitar ingredientes de la receta mientras siga sabiendo bien. |
| **Hiperparámetros** | Las perillas del modelo; se prueban, no se aprenden. |
| **PR-AUC** | Qué tan bien pongo a los tardíos hasta arriba de la lista (azar = % de tardíos). |
| **Sobreajuste** | Memorizar en vez de aprender: muy bien en train, peor en validación. |

**Bonus:** *umbral* = la línea a partir de la cual dices "alerta" · *precisión* = de mis alertas, cuántas eran ciertas · *recall* = de los tardíos, cuántos atrapé · *lift* = cuántas veces mejor que el azar.

---

## 5. Funciones clave (qué notebook las usa)

| Función | Notebook | Qué hace |
|---|---|---|
| `cargar_tablas`, `limpiar_*` | 01 | Leer y arreglar tablas |
| `limpiar_orders`, `definir_objetivo` | 01 | Filtrar pedidos y crear el label |
| `construir_dataset` | 03 | Una fila por pedido con todas las variables |
| `tasa_historica`, `tasa_reciente` | 03 | Historial y carga reciente sin trampa |
| `haversine_km` | 03 | Distancia sobre la Tierra |
| `corte_test`, `folds_temporales` | 04 | Separar por tiempo con purga |
| `ajustar_codificador`, `construir_X` | 04 | One-hot sólo con train |
| `seleccion_rfecv` | 05 | Elegir variables |
| `busqueda_hiperparametros` | 05 | Elegir perillas |
| `evaluar_cv` | 05, 05d | Entrenar y medir en los 5 folds |
| `umbral_optimo` | 05 | Umbral con F1 máximo |
| `RedTabular`, `crear_red` | 05d | La red neuronal |
| `mejor_modelo` | 06 | Regla de selección con empate → simple |
| `captura_top_k` | 06 | % de tardíos atrapados revisando el top *k* % |

---

## 6. Cifras de memoria

| Dato | Valor |
|---|---|
| Pedidos analizados | 96,201 (2017 – ago 2018) |
| Tasa de tardíos | 6.8 % (train 7.6 %, test 4.4 %) |
| Reseña tardío vs. a tiempo | 2.3★ vs. 4.3★ · 62 % vs. 9 % con 1–2★ |
| Variables | 36 base → 105 columnas con one-hot |
| Train / Test / Purgados | 68,287 / 25,350 / 2,564 |
| PR-AUC val (CV) | Log 0.153 · RF 0.158 · XGB 0.157 · Red 0.190 |
| PR-AUC test | Log 0.098 · Red 0.095 (azar 0.044) → ≈ 2.2× |
| Captura top 20 % | 39 % de los retrasos (Logística) |

---

## 7. Preguntas probables del jurado

1. **¿Por qué no usaron accuracy?** Porque con 7 % de tardíos, decir siempre "a tiempo" da 93 % y no sirve. Usamos PR-AUC.
2. **¿Por qué split temporal y no aleatorio?** Porque en la vida real predices el futuro con el pasado, y la tasa de retraso cambia con el tiempo. Un split aleatorio mezclaría el futuro en el entrenamiento.
3. **¿Qué es la purga?** Quitar del entrenamiento los pedidos cuyo resultado aún no se conocía al empezar la validación o el test.
4. **¿Cómo saben que no hay fuga en el historial del vendedor?** Lo recalculamos a mano para 300 pedidos y coincidió en todos.
5. **¿Por qué ganó la logística si la red fue mejor en validación?** Por la regla fijada de antemano entre los clásicos (empate → el más simple). Y en test la red no fue mejor: con desempeño igual, preferimos lo simple y explicable.
6. **¿Por qué los resultados son "solo" 2× el azar?** La fecha prometida ya trae colchón y faltan datos clave (transportista, clima). Aun así, el modelo concentra el 39 % de los retrasos en el 20 % de pedidos.
7. **¿Por qué el umbral marca tantos pedidos en test?** Porque se eligió en validación, cuando había más retrasos. Por eso recomendamos operar por capacidad (revisar el top *k* %).
8. **¿Qué es el sobreajuste y quién lo tuvo?** Memorizar el train. Random Forest y XGBoost: brecha de ROC de 0.20 entre train y validación.
9. **¿Por qué la red no usa RFECV?** Porque RFECV necesita coeficientes o importancias y una red no los tiene. Comparamos listas candidatas de variables.
10. **¿Qué harían después?** Agregar datos de transportista y clima, calibrar probabilidades, predecir los días de retraso y medir el impacto con un experimento A/B.

---

## 8. Autoexamen (tápate las respuestas)

1. ¿Qué columna es el label? → `delivery_late`
2. ¿Qué significa `label_conocida_en`? → El momento en que se supo si llegó tarde.
3. ¿Qué función busca "el último dato anterior a esta fecha"? → `pd.merge_asof`
4. ¿Por qué se suaviza la tasa del vendedor? → Para no exagerar con vendedores de pocos pedidos.
5. ¿Cuántos folds y de qué tipo? → 5, de ventana creciente, con purga.
6. ¿Con qué datos se ajusta el one-hot? → Sólo train.
7. ¿Qué hace RFECV? → Quita las variables menos importantes y se queda con el mejor número.
8. ¿Qué métrica decide al ganador? → PR-AUC promedio en validación.
9. ¿Con qué métrica se ve el sobreajuste y por qué? → ROC-AUC, porque el PR-AUC depende del % de tardíos de cada periodo.
10. ¿Cuántas veces se abre el test? → Una.
