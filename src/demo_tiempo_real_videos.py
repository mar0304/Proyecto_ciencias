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

# --- 1. Voz Sintética Asíncrona ---
def hablar_seguro(texto):
    def ejecutar():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(texto)
            engine.runAndWait()
        except Exception:
            pass

    threading.Thread(target=ejecutar, daemon=True).start()

# --- 2. Visualización de los 21 Puntos Clave de la Mano ---
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anular
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Meñique y palma
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

    for rastro in historial_puntas.values():
        for i in range(1, len(rastro)):
            cv2.line(frame, rastro[i - 1], rastro[i], (255, 200, 0), 2)

    for idx_a, idx_b in CONEXIONES_MANO:
        cv2.line(frame, coords_px[idx_a], coords_px[idx_b], (0, 255, 120), 2)

    for idx, (cx, cy) in enumerate(coords_px):
        if idx == 0:
            color = (0, 0, 255)
            radio = 7
        elif idx in (4, 8, 12, 16, 20):
            color = (0, 255, 255)
            radio = 7
        else:
            color = (255, 100, 0)
            radio = 5

        cv2.circle(frame, (cx, cy), radio, color, -1)
        cv2.circle(frame, (cx, cy), radio + 2, (255, 255, 255), 1)

        etiqueta = NOMBRES_PUNTOS_CLAVE.get(idx, str(idx))
        cv2.putText(frame, etiqueta, (cx + 7, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    return coords_px

# --- 3. Definición y Carga del Modelo LSTM ---
class VideoSequenceLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, num_layers=2, num_classes=4):
        super(VideoSequenceLSTM, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.proj(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

MODELO_PATH = "modelos/modelo_lsch_videos_lstm.pth"
HAND_TASK_PATH = "modelos/hand_landmarker.task"

if not os.path.exists(MODELO_PATH):
    print(f"\n[ERROR] No se encontró '{MODELO_PATH}'. Ejecuta 'python src/entrenar_desde_videos.py' primero.")
    exit()

print("Cargando modelo LSTM...")
checkpoint = torch.load(MODELO_PATH, map_location=torch.device('cpu'))
clases = checkpoint["clases"]
FRAMES_POR_SECUENCIA = checkpoint.get("frames_por_secuencia", 10)
TAMANO_FRAME = tuple(checkpoint.get("tamano_frame", (32, 32)))

modelo = VideoSequenceLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    num_classes=len(clases)
)
modelo.load_state_dict(checkpoint["model_state_dict"])
modelo.eval()
print("¡Modelo LSTM cargado!")

detector_mano = None
if os.path.exists(HAND_TASK_PATH):
    base_options = mp_python.BaseOptions(model_asset_path=HAND_TASK_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    detector_mano = mp_vision.HandLandmarker.create_from_options(options)

buffer_frames = deque(maxlen=FRAMES_POR_SECUENCIA)
historial_puntas = {idx: deque(maxlen=FRAMES_POR_SECUENCIA) for idx in (0, 4, 8, 12, 16, 20)}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

ultima_voz = 0
ultima_prediccion = "Esperando gesto..."
num_puntos_detectados = 0

print("\n=== DEMO LSCh EN TIEMPO REAL CON RED LSTM + PUNTOS VISIBLES ===")

while True:
    exito, frame = cap.read()
    if not exito or frame is None:
        continue

    frame = cv2.flip(frame, 1)

    # Extraer vector de imagen ANTES de dibujar los puntos encima para no alterar la entrada
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_redim = cv2.resize(gris, TAMANO_FRAME)
    vector_frame = (frame_redim.flatten() / 255.0).astype(np.float32)
    buffer_frames.append(vector_frame)

    # Detectar y dibujar los 21 puntos clave de la mano sobre el frame
    if detector_mano is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultados = detector_mano.detect(mp_image)
        if resultados.hand_landmarks:
            mano = resultados.hand_landmarks[0]
            num_puntos_detectados = len(mano)
            coords_px = dibujar_puntos_y_esqueleto(frame, mano, historial_puntas)
            for idx_punta in historial_puntas:
                historial_puntas[idx_punta].append(coords_px[idx_punta])
        else:
            num_puntos_detectados = 0
            for rastro in historial_puntas.values():
                rastro.clear()

    if len(buffer_frames) == FRAMES_POR_SECUENCIA:
        secuencia_array = np.array(list(buffer_frames), dtype=np.float32)  # (10, 1024)
        entrada_tensor = torch.tensor(secuencia_array, dtype=torch.float32).unsqueeze(0)  # (1, 10, 1024)
        
        with torch.no_grad():
            logits = modelo(entrada_tensor)
            probabilidades = torch.softmax(logits, dim=1)[0].numpy()

        max_prob = float(np.max(probabilidades))
        clase_idx = int(np.argmax(probabilidades))
        prediccion = clases[clase_idx]

        # Umbral ajustado al 35% para asegurar la detección en vivo
        if max_prob >= 0.35 and prediccion not in ["nada", "reposo"]:
            ultima_prediccion = f"{prediccion.upper()} ({int(max_prob * 100)}%)"
            
            if (time.time() - ultima_voz) > 2.0:
                pred_clean = prediccion.lower().strip()
                
                if pred_clean == "hola":
                    hablar_seguro("Hola")
                elif "wena" in pred_clean:
                    hablar_seguro("Wena po")
                elif "como" in pred_clean or "estai" in pred_clean:
                    hablar_seguro("¿Cómo estái?")
                
                ultima_voz = time.time()
                buffer_frames.clear()
        else:
            ultima_prediccion = f"En reposo / {prediccion} ({int(max_prob * 100)}%)"

    # Interfaz en pantalla
    cv2.rectangle(frame, (10, 10), (620, 95), (20, 20, 20), -1)
    cv2.putText(frame, f"Traduccion (LSTM): {ultima_prediccion}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
    cv2.putText(frame, f"Puntos identificados: {num_puntos_detectados}/21 | Bufer LSTM: {len(buffer_frames)}/{FRAMES_POR_SECUENCIA}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.imshow('Demo LSCh - Secuencia Temporal LSTM (Puntos Visibles)', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

if detector_mano is not None:
    detector_mano.close()
cap.release()
cv2.destroyAllWindows()