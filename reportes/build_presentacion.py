"""Genera la presentación HTML (autocontenida) con los resultados reales de los notebooks.

Uso (desde la raíz del proyecto):  .venv/bin/python reportes/build_presentacion.py [--pdf]
Requiere haber corrido los notebooks 01–06 (lee reportes/metricas/*.json y modelos/*/best_params.json).
Opcional: reportes/presentacion/powerbi.png (captura del tablero de Power BI) para el PDF y el modo sin internet.
"""
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import config  # noqa: E402

M = config.METRICAS
datos = {
    "eda": json.loads((M / "eda.json").read_text()),
    "split": json.loads((M / "split.json").read_text()),
    "limpieza": json.loads((M / "limpieza.json").read_text()),
    "resultados": json.loads((M / "resultados.json").read_text()),
    "params": {n: json.loads((config.MODELOS / c / "best_params.json").read_text())
               for n, c in config.CLAVES_MODELO.items()},
}
# Variables base por pedido (antes del one-hot)
import pandas as pd  # noqa: E402
from src.preprocesamiento import COLUMNAS_EXCLUIR  # noqa: E402
cols = pd.read_parquet(config.DATOS_PROCESSED / "dataset_pedidos.parquet").columns
datos["n_base"] = int(sum(c not in COLUMNAS_EXCLUIR for c in cols))
# No exportar las curvas de aprendizaje completas dentro del HTML
for r in datos["resultados"]["resumen_modelos"].values():
    r.pop("curvas", None)

carpeta = RAIZ / "reportes" / "presentacion"

# Captura opcional del tablero de Power BI (powerbi.png/.jpg en esta carpeta): se usa en el PDF
# y como respaldo en vivo si no hay internet. El tablero interactivo no necesita este archivo.
import base64  # noqa: E402
for ext, mime in [("png", "image/png"), ("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("webp", "image/webp")]:
    captura = carpeta / f"powerbi.{ext}"
    if captura.exists():
        datos["pbi_poster"] = f"data:{mime};base64," + base64.b64encode(captura.read_bytes()).decode()
        print("Captura del tablero:", captura.name)
        break

plantilla = (carpeta / "plantilla.html").read_text()
html = plantilla.replace("__DATA__", json.dumps(datos, ensure_ascii=False).replace("</", "<\\/"))
salida = carpeta / "presentacion.html"
salida.write_text(html)
print("Presentación:", salida)

if "--pdf" in sys.argv:
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    pdf = carpeta / "presentacion.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--hide-scrollbars",
                    "--virtual-time-budget=8000", f"--print-to-pdf={pdf}", salida.as_uri() + "?print=1"],
                   check=True, capture_output=True)
    print("PDF de respaldo:", pdf)
