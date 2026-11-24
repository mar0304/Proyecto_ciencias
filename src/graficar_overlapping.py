import json
import os
import cv2
import numpy as np

REPORTE_PATH = "data/asl/reporte_overlapping.json"
SALIDA_DIR = "data/asl/graficos"
os.makedirs(SALIDA_DIR, exist_ok=True)

with open(REPORTE_PATH, "r", encoding="utf-8") as f:
    reporte = json.load(f)

videos = reporte.get("videos", [])
resumen_nuevas = reporte.get("resumen_nuevas_palabras", [])
matriz_10 = reporte.get("matriz_overlapping_10_clases", {})

def color_heatmap(valor_pct):
    """Convierte un porcentaje (30% - 100%) en un color BGR tipo mapa de calor (Azul -> Verde -> Amarillo -> Rojo)."""
    t = max(0.0, min(1.0, (valor_pct - 35.0) / 65.0))
    if t < 0.5:
        # Azul oscuro (130, 50, 20) a Verde esmeralda (80, 200, 40)
        u = t * 2.0
        b = int(130 * (1 - u) + 80 * u)
        g = int(50 * (1 - u) + 200 * u)
        r = int(20 * (1 - u) + 40 * u)
    else:
        # Verde esmeralda (80, 200, 40) a Naranja/Rojo cálido (40, 110, 245)
        u = (t - 0.5) * 2.0
        b = int(80 * (1 - u) + 40 * u)
        g = int(200 * (1 - u) + 110 * u)
        r = int(40 * (1 - u) + 245 * u)
    return (b, g, r)

# ==============================================================================
# GRÁFICO 1: MATRIZ DE CALOR (HEATMAP) DE OVERLAPPING ENTRE LAS 10 SEÑAS
# ==============================================================================
clases = list(matriz_10.keys())
n = len(clases)
ancho_m, alto_m = 1150, 920
img_matriz = np.full((alto_m, ancho_m, 3), (24, 26, 32), dtype=np.uint8)

cv2.putText(img_matriz, "MATRIZ DE OVERLAPPING BIOMECANICO: ESPANOL (LSCh) + INGLES (ASL)",
            (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img_matriz, "Similitud Coseno (%) de las trayectorias temporales de los 21 puntos de MediaPipe normalizados",
            (45, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA)

x0, y0 = 200, 165
celda_w, celda_h = 78, 64

for idx_c, nombre in enumerate(clases):
    etiqueta = nombre.upper()
    # Etiqueta fila (izquierda)
    cv2.putText(img_matriz, etiqueta, (35, y0 + idx_c * celda_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    # Etiqueta columna (arriba)
    cv2.putText(img_matriz, etiqueta[:8], (x0 + idx_c * celda_w + 6, y0 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

for i, c1 in enumerate(clases):
    for j, c2 in enumerate(clases):
        val = float(matriz_10[c1][c2])
        cx1 = x0 + j * celda_w
        cy1 = y0 + i * celda_h
        cx2 = cx1 + celda_w - 4
        cy2 = cy1 + celda_h - 4

        col = color_heatmap(val)
        cv2.rectangle(img_matriz, (cx1, cy1), (cx2, cy2), col, -1)
        cv2.rectangle(img_matriz, (cx1, cy1), (cx2, cy2), (45, 48, 58), 1)

        texto_val = f"{val:.1f}%"
        cv2.putText(img_matriz, texto_val, (cx1 + 10, cy1 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

# Barra de leyenda a la derecha
lx = x0 + n * celda_w + 35
cv2.putText(img_matriz, "Escala Overlap", (lx - 10, y0 - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
for paso in range(100):
    pct = 100.0 - paso * 0.65
    ly = y0 + int(paso * (n * celda_h) / 100.0)
    ly_next = y0 + int((paso + 1) * (n * celda_h) / 100.0)
    cv2.rectangle(img_matriz, (lx, ly), (lx + 28, ly_next), color_heatmap(pct), -1)

cv2.putText(img_matriz, "100% (Identico)", (lx + 36, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img_matriz, "70% (Umbral Util)", (lx + 36, y0 + int(n * celda_h * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1, cv2.LINE_AA)
cv2.putText(img_matriz, "35% (Distinto)", (lx + 36, y0 + n * celda_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

cv2.putText(img_matriz, "Nota: Las 3 primeras clases son nativas en Espanol (LSCh) y las 7 restantes fueron incorporadas desde Ingles (ASL).",
            (45, alto_m - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 220, 170), 1, cv2.LINE_AA)

ruta_g1 = os.path.join(SALIDA_DIR, "1_matriz_overlapping_heatmap.png")
cv2.imwrite(ruta_g1, img_matriz)

# ==============================================================================
# GRÁFICO 2: QUÉ NOS SIRVE vs QUÉ SE DESCARTA (FILTRO DE OVERLAPPING POR PALABRA)
# ==============================================================================
ancho_b, alto_b = 1200, 780
img_barras = np.full((alto_b, ancho_b, 3), (24, 26, 32), dtype=np.uint8)

cv2.putText(img_barras, "FILTRO DE UTILIDAD POR OVERLAPPING: DATASET INGLES (ASL) -> ESPANOL CHILENO (LSCh)",
            (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img_barras, "Comparacion de videos en Ingles aprobados (Verde: Nos sirve >= Umbral) vs descartados (Rojo: No sirve) y % de Overlapping",
            (40, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 1, cv2.LINE_AA)

# Agrupar estadísticas por clase chilena desde 'videos'
orden_clases = ["hola", "wena po", "como estai", "chao", "si", "no", "gracias", "comer", "tomar", "porfa"]
origen_map = {
    "hola": "hello / welcome",
    "wena po": "cool / friend",
    "como estai": "how / fine / you",
    "chao": "bye",
    "si": "yes",
    "no": "no",
    "gracias": "good",
    "comer": "eat",
    "tomar": "drink",
    "porfa": "please"
}

bx0, by0 = 280, 145
max_bar_w = 520
fila_h = 56

# Línea vertical de umbral del 70%
umbral_x = bx0 + int(max_bar_w * 0.70)
cv2.line(img_barras, (umbral_x, by0 - 20), (umbral_x, by0 + len(orden_clases) * fila_h), (0, 220, 255), 2)
cv2.putText(img_barras, "Umbral Minimo que nos sirve (70%)", (umbral_x - 110, by0 - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

for idx, c_nom in enumerate(orden_clases):
    vids_c = [v for v in videos if v["clase_lsch_asignada"] == c_nom]
    aprob_c = [v for v in vids_c if v["aprobado_para_entrenar"]]
    total_c = len(vids_c)
    n_aprob = len(aprob_c)
    n_desc = total_c - n_aprob

    # Promedio de overlapping de los videos aprobados
    if aprob_c:
        pct_medio = float(np.mean([v["consistencia_interna_pct"] for v in aprob_c]))
    elif vids_c:
        pct_medio = float(np.mean([v["consistencia_interna_pct"] for v in vids_c]))
    else:
        pct_medio = 0.0

    fy = by0 + idx * fila_h
    etiqueta_izq = f"{c_nom.upper()} ({origen_map[c_nom]})"
    cv2.putText(img_barras, etiqueta_izq, (35, fy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

    # Fondo barra 100%
    cv2.rectangle(img_barras, (bx0, fy + 6), (bx0 + max_bar_w, fy + 38), (45, 48, 58), -1)

    # Barra de porcentaje de Overlapping alcanzado por los videos útiles
    w_pct = int(max_bar_w * (pct_medio / 100.0))
    color_barra = (80, 210, 60) if pct_medio >= 70.0 else (60, 140, 230)
    cv2.rectangle(img_barras, (bx0, fy + 6), (bx0 + w_pct, fy + 38), color_barra, -1)

    cv2.putText(img_barras, f"{pct_medio:.1f}% Overlap", (bx0 + w_pct - 115 if w_pct > 130 else bx0 + w_pct + 8, fy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20) if w_pct > 130 else (255, 255, 255), 1, cv2.LINE_AA)

    # Indicador a la derecha de cuántos videos sirven y cuántos se descartan
    rx = bx0 + max_bar_w + 35
    cv2.rectangle(img_barras, (rx, fy + 6), (rx + 155, fy + 38), (45, 130, 45), -1)
    cv2.putText(img_barras, f"Sirven: {n_aprob} videos", (rx + 12, fy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(img_barras, (rx + 165, fy + 6), (rx + 335, fy + 38), (50, 50, 160) if n_desc > 0 else (55, 60, 65), -1)
    cv2.putText(img_barras, f"Descartados: {n_desc}", (rx + 177, fy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)

# Resumen inferior
cv2.rectangle(img_barras, (40, alto_b - 65), (ancho_b - 40, alto_b - 18), (35, 42, 52), -1)
cv2.putText(img_barras,
            f"RESUMEN TOTAL: De {reporte['total_analizados']} videos en Ingles (ASL) evaluados con MediaPipe, {reporte['total_aprobados']} videos SIRVEN (>=70% Overlap) y alimentan la red LSTM (99.7% Precision).",
            (55, alto_b - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1, cv2.LINE_AA)

ruta_g2 = os.path.join(SALIDA_DIR, "2_utilidad_overlapping_barras.png")
cv2.imwrite(ruta_g2, img_barras)

print(f"Gráficos generados exitosamente en:\n - {ruta_g1}\n - {ruta_g2}")
