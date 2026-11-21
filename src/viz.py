"""Estilo de gráficas común a todos los notebooks (paleta validada, marcas finas, ejes discretos)."""
import matplotlib.pyplot as plt

# Un color fijo por entidad (nunca por ranking)
COLOR_MODELO = {
    "Regresión Logística": "#2a78d6",   # azul
    "Random Forest": "#eb6834",         # naranja
    "XGBoost": "#1baf7a",               # aqua
    "Red Neuronal": "#eda100",          # amarillo
}
TARDIO = "#e34948"      # rojo: clase "tardío"
A_TIEMPO = "#2a78d6"    # azul: clase "a tiempo"
NEUTRO = "#898781"      # gris: referencias (azar, cortes)
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRID = "#e1e0d9"
EJE = "#c3c2b7"
SUPERFICIE = "#fcfcfb"


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
        "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def guardar(fig, nombre):
    from . import config
    fig.savefig(config.FIGURAS / f"{nombre}.png")
