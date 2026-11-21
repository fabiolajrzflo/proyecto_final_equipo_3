"""Convierte las guías de docs/*.md a PDF (Markdown -> HTML con estilos de impresión -> PDF con Chrome).

Uso (desde la raíz del proyecto):  .venv/bin/python docs/build_pdf.py
"""
import re
import subprocess
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

DOCS = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
GUIAS = {
    "guia_detallada_notebooks": "Guía detallada del código",
    "guia_rapida_notebooks": "Guía rápida para estudiar",
}

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
        @bottom-right { content: counter(page) " / " counter(pages); font: 9pt "Helvetica Neue", Arial, sans-serif; color: #898781; }
        @bottom-left { content: "Proyecto Olist · Equipo 3"; font: 9pt "Helvetica Neue", Arial, sans-serif; color: #898781; } }
:root { --navy: #051c2c; --azul: #2a78d6; --rojo: #e34948; --gris: #52514e; --linea: #e1e0d9; --fondo: #f5f4f0; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.4pt; line-height: 1.5; color: #1a1a1a; margin: 0; }
h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 400; font-size: 25pt; color: var(--navy); line-height: 1.15;
     margin: 0 0 10pt; padding-bottom: 8pt; border-bottom: 3px solid var(--navy); }
h2 { font-family: Georgia, "Times New Roman", serif; font-weight: 400; font-size: 17pt; color: var(--navy);
     margin: 20pt 0 8pt; padding-bottom: 4pt; border-bottom: 1px solid var(--linea); break-after: avoid; }
h3 { font-size: 12pt; color: var(--azul); margin: 14pt 0 5pt; break-after: avoid; }
h4 { font-size: 11pt; margin: 10pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; }
a { color: var(--azul); text-decoration: none; }
strong { color: #000; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; } li { margin-bottom: 3pt; }
blockquote { margin: 8pt 0 10pt; padding: 8pt 12pt; background: var(--fondo); border-left: 4px solid var(--azul); color: #222; break-inside: avoid; }
blockquote p { margin: 0 0 4pt; } blockquote p:last-child { margin: 0; }
hr { border: 0; border-top: 1px solid var(--linea); margin: 14pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.2pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: var(--navy); color: #fff; text-align: left; padding: 5pt 7pt; font-weight: 600; }
td { padding: 4.5pt 7pt; border-bottom: 1px solid var(--linea); vertical-align: top; }
tbody tr:nth-child(even) td { background: #fafaf8; }
code { font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 8.6pt; background: #eef2f7; padding: 1pt 3pt; border-radius: 3px; color: #0b2e45; }
pre, .codehilite { background: #f6f8fa; border: 1px solid var(--linea); border-left: 4px solid var(--navy); border-radius: 3px;
       padding: 8pt 10pt; margin: 6pt 0 10pt; break-inside: avoid; }
pre code, .codehilite code { background: none; padding: 0; font-size: 8.3pt; line-height: 1.45; white-space: pre-wrap; word-break: break-word; color: inherit; }
.codehilite pre { margin: 0; padding: 0; border: 0; background: none; }
.portada { margin-bottom: 14pt; }
.portada .kicker { font-size: 9pt; letter-spacing: .16em; text-transform: uppercase; color: var(--azul); font-weight: 700; margin-bottom: 6pt; }
/* La guía detallada: cada notebook y cada anexo empieza en página nueva */
.detallada h2.nueva { break-before: page; }
"""


LISTA = re.compile(r"^( *)([-*]|\d+\.)\s")


def normalizar_markdown(texto):
    """Adapta el Markdown estilo GitHub/VS Code a Python-Markdown:
    - línea en blanco antes de cada lista,
    - sublistas con 4 espacios (en el .md usan 2),
    - bloques de código dentro de listas se sacan al nivel principal."""
    out, en_codigo, sangria_codigo = [], False, 0
    for linea in texto.split("\n"):
        sin_sangria = linea.lstrip(" ")
        sangria = len(linea) - len(sin_sangria)
        if sin_sangria.startswith("```"):
            if not en_codigo:
                en_codigo, sangria_codigo = True, sangria
                if out and out[-1].strip():
                    out.append("")
            else:
                en_codigo = False
            out.append(sin_sangria)
            continue
        if en_codigo:
            out.append(linea[sangria_codigo:] if linea.startswith(" " * sangria_codigo) else linea)
            continue
        m = LISTA.match(linea)
        if m:
            n = len(m.group(1))
            linea = " " * (2 * n) + linea[n:]
            previa = out[-1] if out else ""
            if previa.strip() and not LISTA.match(previa):
                out.append("")
        out.append(linea)
    return "\n".join(out)


def construir(nombre, titulo):
    md_texto = normalizar_markdown((DOCS / f"{nombre}.md").read_text())
    html_cuerpo = markdown.markdown(
        md_texto,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False, "css_class": "codehilite"},
                           "toc": {"slugify": _slug}},
    )
    # Enlaces entre guías: apuntar a los PDF
    html_cuerpo = html_cuerpo.replace(".md\"", ".pdf\"").replace(".md#", ".pdf#")
    clase = "detallada" if "detallada" in nombre else "rapida"
    if clase == "detallada":
        html_cuerpo = re.sub(r"<h2 id=\"(notebook|anexo)", r'<h2 class="nueva" id="\1', html_cuerpo)
    pyg = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{titulo}</title>
<style>{CSS}\n{pyg}\n.codehilite {{ background: #f6f8fa; }}</style></head>
<body class="{clase}"><div class="portada"><div class="kicker">Proyecto final · Predicción de entregas tardías · Olist</div></div>
{html_cuerpo}</body></html>"""
    ruta_html = DOCS / f".{nombre}.html"
    ruta_html.write_text(html)
    pdf = DOCS / f"{nombre}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", ruta_html.as_uri()], check=True, capture_output=True)
    ruta_html.unlink()
    print("PDF:", pdf)


def _slug(valor, separador):
    """Mismo formato de anclas que GitHub/VS Code (para que funcionen los enlaces del índice)."""
    valor = valor.strip().lower()
    valor = re.sub(r"[^\w\s-]", "", valor, flags=re.UNICODE)
    return re.sub(r"\s", separador, valor)


if __name__ == "__main__":
    for nombre, titulo in GUIAS.items():
        construir(nombre, titulo)
