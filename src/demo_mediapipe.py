import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyttsx3
import threading
import time
from collections import deque
import os

def hablar_async(texto):
    def ejecutar():
        try:
            engine = pyttsx3.init()
            engine.say(texto)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=ejecutar, daemon=True).start()

# --- Definición de la Red Neuronal LSTM ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, num_layers=2, num_classes=11):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# --- Instructivo de las 10 Señas Chilenas Disponibles ---
INSTRUCTIVO_SENAS = [
    ("HOLA",       "Mano abierta saludando a la altura de la cara",        "Hola"),
    ("WENA PO",    "Saludo chileno levantando o agitando la mano",         "Wena po"),
    ("COMO ESTAI", "Mano frente al pecho preguntando como estas",          "¿Cómo estái?"),
    ("CHAO",       "Mano abierta moviendose de lado a lado (despedida)",   "Chao, nos vemos"),
    ("SI",         "Puno cerrado subiendo y bajando (como asintiendo)",    "Sí po"),
    ("NO",         "Juntar dedos indice y medio contra el pulgar",         "No pasa nada"),
    ("GRACIAS",    "Mano plana desde el menton/boca hacia adelante",       "Muchas gracias"),
    ("COMER",      "Puntas de los dedos unidas acercandose a la boca",     "Vamos a comer"),
    ("TOMAR",      "Mano en forma de 'C' (vaso) inclinada hacia la boca",  "Quiero tomar agua"),
    ("PORFA",      "Mano abierta haciendo un circulo sobre el pecho",      "Por favor"),
]

FRASES_VOZ_CHILENA = {nombre.lower(): voz for nombre, _, voz in INSTRUCTIVO_SENAS}

# --- Conexiones y Etiquetas de los 21 Puntos Clave de la Mano ---
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),                  # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),                  # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),             # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),           # Anular
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Meñique y palma
]

NOMBRES_PUNTOS_CLAVE = {
    0: "0:Muneca",
    4: "4:Pulgar",
    8: "8:Indice",
    12: "12:Medio",
    16: "16:Anular",
    20: "20:Menique"
}

def dibujar_puntos_y_esqueleto(frame, mano_landmarks, historial_puntas):
    alto, ancho, _ = frame.shape
    coords_px = []

    for lm in mano_landmarks:
        cx, cy = int(lm.x * ancho), int(lm.y * alto)
        coords_px.append((cx, cy))

    # 1. Dibujar rastro temporal (estela de movimiento que analiza la LSTM)
    for rastro in historial_puntas.values():
        for i in range(1, len(rastro)):
            cv2.line(frame, rastro[i - 1], rastro[i], (255, 200, 0), 2, cv2.LINE_AA)

    # 2. Dibujar conexiones óseas entre articulaciones
    for idx_a, idx_b in CONEXIONES_MANO:
        cv2.line(frame, coords_px[idx_a], coords_px[idx_b], (0, 255, 120), 2, cv2.LINE_AA)

    # 3. Dibujar los 21 puntos visibles con código de color e identificador
    for idx, (cx, cy) in enumerate(coords_px):
        if idx == 0:
            color = (0, 0, 255)    # Muñeca (Rojo)
            radio = 7
        elif idx in (4, 8, 12, 16, 20):
            color = (0, 255, 255)  # Puntas de los dedos (Amarillo brillante)
            radio = 7
        else:
            color = (255, 100, 0)  # Articulaciones intermedias (Azul/Naranja)
            radio = 5

        cv2.circle(frame, (cx, cy), radio, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radio + 2, (255, 255, 255), 1, cv2.LINE_AA)

        etiqueta = NOMBRES_PUNTOS_CLAVE.get(idx, str(idx))
        cv2.putText(frame, etiqueta, (cx + 7, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return coords_px

def construir_panel_instructivo(alto_frame, prediccion_actual):
    """Construye un panel lateral con el instructivo de palabras y cómo hacer cada seña."""
    ancho_panel = 470
    panel = np.full((alto_frame, ancho_panel, 3), (26, 28, 36), dtype=np.uint8)

    # Encabezado del panel
    cv2.rectangle(panel, (0, 0), (ancho_panel, 72), (38, 44, 58), -1)
    cv2.putText(panel, "INSTRUCTIVO DE SENAS (LSCh + LSTM)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(panel, "Haz cualquiera de estos gestos frente a la camara:", (16, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 220, 230), 1, cv2.LINE_AA)

    y_base = 85
    espacio_y = max(36, (alto_frame - 135) // len(INSTRUCTIVO_SENAS))

    pred_limpia = prediccion_actual.lower().split("(")[0].strip()

    for idx, (palabra, como_hacerla, frase_voz) in enumerate(INSTRUCTIVO_SENAS):
        y_top = y_base + idx * espacio_y
        y_bot = y_top + espacio_y - 4

        es_activa = (palabra.lower() == pred_limpia)
        color_fondo = (40, 110, 50) if es_activa else (34, 38, 48)
        color_borde = (0, 255, 120) if es_activa else (55, 60, 75)

        cv2.rectangle(panel, (12, y_top), (ancho_panel - 12, y_bot), color_fondo, -1)
        cv2.rectangle(panel, (12, y_top), (ancho_panel - 12, y_bot), color_borde, 1)

        # Nombre de la palabra y lo que dice por voz
        cv2.putText(panel, f"{idx + 1}. {palabra}", (20, y_top + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255) if not es_activa else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Voz: "{frase_voz}"', (175, y_top + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 230, 160), 1, cv2.LINE_AA)

        # Explicación de cómo hacer la seña
        cv2.putText(panel, f"Gesto: {como_hacerla}", (20, y_top + 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (225, 225, 225), 1, cv2.LINE_AA)

    # Pie del panel con controles y leyenda de colores de los 21 puntos
    cv2.rectangle(panel, (0, alto_frame - 44), (ancho_panel, alto_frame), (18, 20, 25), -1)
    cv2.circle(panel, (22, alto_frame - 22), 5, (0, 0, 255), -1)
    cv2.putText(panel, "Muneca(0)", (32, alto_frame - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(panel, (118, alto_frame - 22), 5, (0, 255, 255), -1)
    cv2.putText(panel, "Puntas(4,8,12,16,20)", (128, alto_frame - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "[I] Ocultar Guia | [ESC] Salir", (285, alto_frame - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 220, 255), 1, cv2.LINE_AA)

    return panel

MODELO_PATH = "modelos/modelo_lsch_mediapipe_lstm.pth"
HAND_TASK_PATH = "modelos/hand_landmarker.task"

if not os.path.exists(MODELO_PATH):
    print(f"\n[ERROR] No se encontró '{MODELO_PATH}'. Ejecuta 'python src/entrenar_mediapipe.py' primero.")
    exit()

checkpoint = torch.load(MODELO_PATH, map_location=torch.device('cpu'))
clases = checkpoint["clases"]
FRAMES_POR_SECUENCIA = checkpoint.get("frames_por_secuencia", 10)
USAR_NORM_MUNECA = checkpoint.get("normalizado_muneca", False)

def normalizar_puntos_mano(mano_landmarks):
    coords = np.array([[lm.x, lm.y] for lm in mano_landmarks], dtype=np.float32)
    if not USAR_NORM_MUNECA:
        return coords.flatten()
    muneca = coords[0].copy()
    relativos = coords[1:] - muneca
    max_dist = np.max(np.linalg.norm(relativos, axis=1))
    if max_dist > 1e-5:
        relativos = relativos / max_dist
    return np.concatenate([muneca, relativos.flatten()]).astype(np.float32)

modelo = SignLanguageLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    num_classes=len(clases)
)
modelo.load_state_dict(checkpoint["model_state_dict"])
modelo.eval()

buffer_puntos = deque(maxlen=FRAMES_POR_SECUENCIA)
historial_puntas = {idx: deque(maxlen=FRAMES_POR_SECUENCIA) for idx in (0, 4, 8, 12, 16, 20)}

base_options = mp_python.BaseOptions(model_asset_path=HAND_TASK_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
detector = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

ultima_voz = 0
prediccion_texto = "Esperando mano..."
palabra_detectada_activa = ""
num_puntos_detectados = 0
mostrar_instructivo = True

print("\n=====================================================================")
print("   INSTRUCTIVO DE SEÑAS CHILENAS (RED NEURONAL LSTM + MEDIAPIPE)")
print("=====================================================================")
for idx, (pal, gesto, voz) in enumerate(INSTRUCTIVO_SENAS, 1):
    print(f" {idx:2d}. {pal:<11} -> Gesto: {gesto:<52} | Voz: '{voz}'")
print("---------------------------------------------------------------------")
print(" Controles en ventana: [I] Mostrar/Ocultar Instructivo | [ESC] Salir")
print("=====================================================================\n")

while True:
    exito, frame = cap.read()
    if not exito:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    resultados = detector.detect(mp_image)

    puntos_frame = np.zeros(42, dtype=np.float32)
    if resultados.hand_landmarks:
        mano = resultados.hand_landmarks[0]
        num_puntos_detectados = len(mano)
        puntos_frame = normalizar_puntos_mano(mano)
        buffer_puntos.append(puntos_frame)

        coords_px = dibujar_puntos_y_esqueleto(frame, mano, historial_puntas)
        for idx_punta in historial_puntas:
            historial_puntas[idx_punta].append(coords_px[idx_punta])
    else:
        num_puntos_detectados = 0
        palabra_detectada_activa = ""
        buffer_puntos.clear()
        for rastro in historial_puntas.values():
            rastro.clear()

    if len(buffer_puntos) == FRAMES_POR_SECUENCIA:
        secuencia_array = np.array(list(buffer_puntos), dtype=np.float32)  # (10, 42)
        
        if np.sum(np.abs(secuencia_array)) > 0:
            entrada_tensor = torch.tensor(secuencia_array, dtype=torch.float32).unsqueeze(0)  # (1, 10, 42)
            with torch.no_grad():
                logits = modelo(entrada_tensor)
                probabilidades = torch.softmax(logits, dim=1)[0].numpy()

            max_prob = float(np.max(probabilidades))
            prediccion = clases[int(np.argmax(probabilidades))]

            if max_prob >= 0.55 and prediccion != "nada":
                prediccion_texto = f"{prediccion.upper()} ({int(max_prob*100)}%)"
                palabra_detectada_activa = prediccion
                
                if (time.time() - ultima_voz) > 2.5:
                    frase = FRASES_VOZ_CHILENA.get(prediccion.lower().strip(), prediccion)
                    hablar_async(frase)
                    
                    ultima_voz = time.time()
                    buffer_puntos.clear()
            else:
                prediccion_texto = f"Analizando... ({prediccion}: {int(max_prob*100)}%)"
    elif num_puntos_detectados == 0:
        prediccion_texto = "No se detecta mano en pantalla"
    else:
        prediccion_texto = f"Capturando gesto... ({len(buffer_puntos)}/{FRAMES_POR_SECUENCIA})"

    # Panel superior informativo sobre la cámara
    cv2.rectangle(frame, (10, 10), (630, 95), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (630, 95), (0, 255, 120), 1)
    cv2.putText(frame, f"Traduccion (LSTM): {prediccion_texto}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Puntos identificados: {num_puntos_detectados}/21 | Secuencia LSTM: {len(buffer_puntos)}/{FRAMES_POR_SECUENCIA}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # Unir el video de la cámara con el panel lateral de instructivo
    if mostrar_instructivo:
        panel_guia = construir_panel_instructivo(frame.shape[0], palabra_detectada_activa)
        vista_final = np.hstack((frame, panel_guia))
    else:
        vista_final = frame

    cv2.imshow("Traductor LSCh con MediaPipe + LSTM (21 Puntos Visibles + Instructivo)", vista_final)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:  # ESC
        break
    elif tecla in (ord('i'), ord('I')):
        mostrar_instructivo = not mostrar_instructivo

detector.close()
cap.release()
cv2.destroyAllWindows()
