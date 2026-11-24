import cv2
import numpy as np
import os
import json
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DATASET_LSCH = "dataset_lsch"
DATASET_ASL = "data/asl/videos"
HAND_TASK_PATH = "modelos/hand_landmarker.task"
REPORTE_PATH = "data/asl/reporte_overlapping.json"
FRAMES_POR_SECUENCIA = 10
NUM_FEATURES = 42  # 21 puntos * (x, y) normalizados respecto a la muñeca

# 1. Mapeo de refuerzo para las 3 señas chilenas originales
MAPA_REFUERZO_LSCH = {
    "hola": ["hello", "welcome"],
    "wena po": ["cool", "friend"],
    "como estai": ["how", "you", "fine", "what"]
}

# 2. Mapeo de 7 NUEVAS palabras en español chileno creadas a partir de videos en inglés (ASL)
#    cuyos gestos comparten alta similitud biomecánica (overlapping universal / cognados gestuales):
NUEVAS_CLASES_CHILENAS_DESDE_ASL = {
    "chao": {
        "asl_word": "bye",
        "voz_chilena": "Chao, nos vemos",
        "descripcion_gesto": "Mano abierta moviéndose de lado a lado en despedida"
    },
    "si": {
        "asl_word": "yes",
        "voz_chilena": "Sí po",
        "descripcion_gesto": "Puño cerrado asintiendo hacia arriba y abajo"
    },
    "no": {
        "asl_word": "no",
        "voz_chilena": "No pasa nada",
        "descripcion_gesto": "Dedos índice y medio cerrándose contra el pulgar"
    },
    "gracias": {
        "asl_word": "good",
        "voz_chilena": "Muchas gracias",
        "descripcion_gesto": "Mano plana saliendo desde el mentón/boca hacia adelante"
    },
    "comer": {
        "asl_word": "eat",
        "voz_chilena": "Vamos a comer",
        "descripcion_gesto": "Puntas de los dedos unidas acercándose a la boca"
    },
    "tomar": {
        "asl_word": "drink",
        "voz_chilena": "Quiero tomar agua",
        "descripcion_gesto": "Mano en forma de C simulando un vaso hacia la boca"
    },
    "porfa": {
        "asl_word": "please",
        "voz_chilena": "Por favor",
        "descripcion_gesto": "Mano abierta haciendo un círculo sobre el pecho"
    }
}

def crear_detector():
    base_options = mp_python.BaseOptions(model_asset_path=HAND_TASK_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def normalizar_puntos_mano(mano_landmarks):
    coords = np.array([[lm.x, lm.y] for lm in mano_landmarks], dtype=np.float32)  # (21, 2)
    muneca = coords[0].copy()
    relativos = coords - muneca
    max_dist = np.max(np.linalg.norm(relativos, axis=1))
    if max_dist > 1e-5:
        relativos = relativos / max_dist
    return relativos.flatten()  # (42,)

def extraer_secuencia_video(ruta_video, detector, num_frames=FRAMES_POR_SECUENCIA):
    cap = cv2.VideoCapture(ruta_video)
    frames_validos = []

    while cap.isOpened():
        exito, frame = cap.read()
        if not exito:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = detector.detect(mp_image)
        if res.hand_landmarks:
            vec = normalizar_puntos_mano(res.hand_landmarks[0])
            frames_validos.append(vec)

    cap.release()
    if len(frames_validos) < 3:
        return None

    indices = np.linspace(0, len(frames_validos) - 1, num_frames, dtype=int)
    secuencia = np.array([frames_validos[i] for i in indices], dtype=np.float32)  # (10, 42)
    return secuencia

def similitud_coseno_secuencia(seq_a, seq_b):
    va = seq_a.flatten()
    vb = seq_b.flatten()
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    cos_sim = float(np.dot(va, vb) / (norm_a * norm_b))
    return max(0.0, min(100.0, (cos_sim + 1.0) / 2.0 * 100.0))

if __name__ == "__main__":
    print("=== ANÁLISIS DE OVERLAPPING Y EXPANSIÓN DE VOCABULARIO CHILENO (LSCh + ASL) ===")
    detector = crear_detector()

    # 1. Extraer secuencias y centroides de todas las palabras descargadas de ASL y de LSCh
    prototipos_totales = {}
    secuencias_por_clase = {}

    for clase_lsch in ["hola", "wena po", "como estai"]:
        carpeta = os.path.join(DATASET_LSCH, clase_lsch)
        if not os.path.exists(carpeta):
            continue
        seqs = []
        for f in os.listdir(carpeta):
            if f.endswith((".mp4", ".avi", ".mov")):
                s = extraer_secuencia_video(os.path.join(carpeta, f), detector)
                if s is not None:
                    seqs.append(s)
        if seqs:
            prototipos_totales[clase_lsch] = np.mean(seqs, axis=0)
            secuencias_por_clase[clase_lsch] = seqs
            print(f"[Base LSCh] '{clase_lsch}': {len(seqs)} videos válidos.")

    resultados_videos = []

    # 2. Analizar las 7 NUEVAS palabras chilenas creadas desde los videos en inglés (ASL)
    print("\n--- ANALIZANDO CONSISTENCIA INTERNA Y OVERLAPPING DE LAS 7 NUEVAS PALABRAS CHILENAS ---")
    resumen_nuevas = []
    for nombre_chileno, info in NUEVAS_CLASES_CHILENAS_DESDE_ASL.items():
        asl_word = info["asl_word"]
        carpeta_asl = os.path.join(DATASET_ASL, asl_word)
        if not os.path.exists(carpeta_asl):
            continue

        archivos = sorted([f for f in os.listdir(carpeta_asl) if f.endswith(".mp4")])
        seqs_palabra = []
        rutas_palabra = []
        for archivo in archivos:
            ruta = os.path.join(carpeta_asl, archivo)
            s = extraer_secuencia_video(ruta, detector)
            if s is not None:
                seqs_palabra.append((archivo, ruta, s))

        if not seqs_palabra:
            continue

        # Centroide de la nueva palabra chilena
        centroide = np.mean([x[2] for x in seqs_palabra], axis=0)
        prototipos_totales[nombre_chileno] = centroide

        # Medir consistencia intra-clase y máximo overlapping con las clases existentes (para evitar colisión/confusión)
        sims_internas = []
        aprobados_clase = 0
        for archivo, ruta, seq in seqs_palabra:
            sim_centroide = similitud_coseno_secuencia(seq, centroide)
            sims_internas.append(sim_centroide)

            # Comparar contra las 3 clases originales para ver si se solapa peligrosamente (>92%)
            sims_contra_otras = {
                c_orig: round(similitud_coseno_secuencia(seq, prototipos_totales[c_orig]), 2)
                for c_orig in ["hola", "wena po", "como estai"] if c_orig in prototipos_totales
            }
            max_overlap_ajeno = max(sims_contra_otras.values()) if sims_contra_otras else 0.0

            # Se aprueba si es consistente con su propio gesto (>= 65%) y no es un clon exacto de otra seña (< 94%)
            aprobado = bool(sim_centroide >= 65.0 and max_overlap_ajeno < 94.0)
            if aprobado:
                aprobados_clase += 1

            resultados_videos.append({
                "tipo": "nueva_clase_chilena",
                "palabra_asl": asl_word,
                "clase_lsch_asignada": nombre_chileno,
                "voz_chilena": info["voz_chilena"],
                "archivo": archivo,
                "ruta": ruta,
                "consistencia_interna_pct": round(sim_centroide, 2),
                "max_overlapping_con_otras_clases_pct": round(max_overlap_ajeno, 2),
                "aprobado_para_entrenar": aprobado
            })

        resumen_nuevas.append({
            "Palabra Chilena": nombre_chileno.upper(),
            "Origen ASL (Inglés)": asl_word,
            "Frase de Voz": info["voz_chilena"],
            "Videos con Mano": len(seqs_palabra),
            "Videos Aprobados": aprobados_clase,
            "Consistencia Gestual (%)": round(float(np.mean(sims_internas)), 1)
        })

    # 3. Analizar videos ASL de refuerzo para las 3 clases originales (hola, wena po, como estai)
    for clase_lsch, lista_asl in MAPA_REFUERZO_LSCH.items():
        if clase_lsch not in prototipos_totales:
            continue
        proto = prototipos_totales[clase_lsch]
        for asl_word in lista_asl:
            carpeta_asl = os.path.join(DATASET_ASL, asl_word)
            if not os.path.exists(carpeta_asl):
                continue
            for archivo in sorted(os.listdir(carpeta_asl)):
                if not archivo.endswith(".mp4"):
                    continue
                ruta = os.path.join(carpeta_asl, archivo)
                seq = extraer_secuencia_video(ruta, detector)
                if seq is None:
                    continue
                sim = similitud_coseno_secuencia(seq, proto)
                resultados_videos.append({
                    "tipo": "refuerzo_clase_existente",
                    "palabra_asl": asl_word,
                    "clase_lsch_asignada": clase_lsch,
                    "voz_chilena": clase_lsch,
                    "archivo": archivo,
                    "ruta": ruta,
                    "consistencia_interna_pct": round(sim, 2),
                    "max_overlapping_con_otras_clases_pct": round(sim, 2),
                    "aprobado_para_entrenar": bool(sim >= 70.0)
                })

    detector.close()

    # 4. Calcular Matriz de Overlapping Inter-Clase (10x10 entre todas las palabras chilenas)
    nombres_clases = list(prototipos_totales.keys())
    matriz_inter = {c1: {} for c1 in nombres_clases}
    for c1 in nombres_clases:
        for c2 in nombres_clases:
            matriz_inter[c1][c2] = round(similitud_coseno_secuencia(prototipos_totales[c1], prototipos_totales[c2]), 1)

    print("\n=== TABLA DE LAS 7 NUEVAS PALABRAS EN CHILENO AGREGADAS DESDE VIDEOS EN INGLÉS ===")
    df_nuevas = pd.DataFrame(resumen_nuevas)
    print(df_nuevas.to_string(index=False))

    print("\n=== MATRIZ DE OVERLAPPING ENTRE LAS 10 PALABRAS CHILENAS (% Similitud entre gestos) ===")
    df_inter = pd.DataFrame(matriz_inter)
    print(df_inter.to_string())

    aprobados_totales = [r for r in resultados_videos if r["aprobado_para_entrenar"]]
    print(f"\nTotal videos ASL analizados con MediaPipe: {len(resultados_videos)}")
    print(f"Total videos ASL aprobados para entrenar la LSTM: {len(aprobados_totales)}")

    reporte = {
        "nuevas_clases_chilenas": NUEVAS_CLASES_CHILENAS_DESDE_ASL,
        "resumen_nuevas_palabras": resumen_nuevas,
        "matriz_overlapping_10_clases": matriz_inter,
        "total_analizados": len(resultados_videos),
        "total_aprobados": len(aprobados_totales),
        "videos": resultados_videos
    }
    with open(REPORTE_PATH, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"\nReporte guardado en '{REPORTE_PATH}'.")
