# Predicción de entregas tardías · Olist

Proyecto final · Equipo 3 · Diplomado IA Generativa, Business Intelligence y Ciencia de Datos

Integrantes de Equipo 3:
Fabiola Juárez Flores
Martha Jessica Campos Aldape
Candy del Carmen Cortes Castro

Clasificación binaria: estimar, **al momento de la compra**, la probabilidad de que un pedido de Olist llegue después de la fecha prometida.

## Flujo (diagrama del profesor)

```
DATA → 01 Limpieza → 02 Exploración → 03 Variables → 04 Split temporal (5 folds + purga + test aparte)
                                                          │
            ┌───────────────────┬─────────────────────────┼──────────────────┐
     05a Logística        05b Random Forest          05c XGBoost       05d Red neuronal
     (RFECV → búsqueda)   (RFECV → búsqueda)         (RFECV → búsqueda) (conjuntos → búsqueda)
            └───────────────────┴─────────────────────────┴──────────────────┘
                                                          │
                        06 Selección con CV → prueba final en test (una sola vez)
```

## Estructura

```
datos/                    CSV originales de Olist (no se modifican)
  interim/                tablas limpias (01)
  processed/              dataset por pedido, X_train/X_test, folds.json (03, 04)
src/                      código reutilizable que importan los notebooks
  config.py               rutas, semilla, fecha de corte del test, decisiones del equipo
  limpieza.py · features.py · splits.py · preprocesamiento.py
  modelos.py · red_neuronal.py · evaluacion.py · viz.py
notebooks/
  01_limpieza … 06_seleccion_y_test_final   (correr en orden)
modelos/<modelo>/         variables, hiperparámetros, CV por fold, predicciones fuera de fold, modelo
reportes/
  figuras/ · metricas/    salidas de los notebooks
  presentacion/presentacion.html   presentación (abrir en el navegador)
  articulo/reporte_tecnico.pdf     reporte técnico (artículo corto)
  build_presentacion.py · build_articulo.py   regeneran ambos con los resultados actuales
```

## Cómo correrlo

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# macOS: XGBoost necesita OpenMP ->  brew install libomp

cd notebooks
for nb in 01_limpieza 02_eda 03_feature_engineering 04_preparacion_dataset \
          05a_modelo_logistica 05b_modelo_random_forest 05c_modelo_xgboost \
          05d_modelo_red_neuronal 06_seleccion_y_test_final; do
  ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
cd ..
.venv/bin/python reportes/build_presentacion.py --pdf
.venv/bin/python reportes/build_articulo.py
```

En VS Code: seleccionar el intérprete `.venv` como kernel de los notebooks.

## Presentación

Abrir `reportes/presentacion/presentacion.html` en Chrome.

| Tecla              | Acción                                   |
| ------------------ | ---------------------------------------- |
| → / espacio / clic | siguiente (revela elementos paso a paso) |
| ←                  | anterior                                 |
| F                  | pantalla completa                        |
| N                  | notas del orador                         |
| T                  | cronómetro (20 min)                      |

`presentacion.pdf` es un respaldo estático por si falla el equipo de la sala.

**Tablero de Power BI (lámina 6).** Se muestra en vivo dentro de la presentación:

- Necesita internet y una sesión de Power BI ya iniciada en el mismo navegador (Chrome o Edge; Safari suele bloquear el inicio de sesión dentro del iframe).
- Se empieza a cargar dos láminas antes, para que ya esté listo al llegar.
- Al hacer clic dentro del tablero, el teclado controla el tablero. Para seguir navegando, haz clic fuera de él (por ejemplo, en el título) y usa →.
- Los botones de la lámina permiten recargarlo, verlo a pantalla completa o abrirlo en una pestaña.
- Opcional: guarda una captura del tablero como `reportes/presentacion/powerbi.png` y vuelve a correr `build_presentacion.py --pdf`. Esa captura aparece en el PDF y se usa como respaldo si no hay internet.
- La URL del reporte está en `reportes/presentacion/plantilla.html`, en el atributo `data-src` del bloque `.pbi`.

## Decisiones de diseño clave

- **Variables point-in-time:** el historial del vendedor/ruta y la carga reciente sólo usan labels conocidos antes de la compra (`label_conocida_en`).
- **Split temporal con purga:** un pedido sólo entrena si su label ya se conocía al inicio de la validación/test.
- **Test intocable:** variables, hiperparámetros, umbral y modelo se eligen con los 5 folds; el test se abre una vez en el notebook 06.
- **Métrica:** PR-AUC (clase desbalanceada). Sobreajuste diagnosticado con la brecha de ROC-AUC train–validación.
- **Regla de selección:** mayor PR-AUC en validación; empate (< 0.005) → el modelo más simple.
- Pedidos sin fecha de entrega: **excluidos** (decisión del equipo).
