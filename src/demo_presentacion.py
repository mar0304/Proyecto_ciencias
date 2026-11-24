import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
import pyttsx3
import threading
import time
import os

# --- 1. Configuración de Voz ---
def hablar_async(texto):
    def ejecutar():
        try:
            motor = pyttsx3.init()
            motor.say(texto)
            motor.runAndWait()
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

def dibujar_puntos_mano(frame, mano_landmarks):
    alto, ancho, _ = frame.shape
    coords_px = [(int(lm.x * ancho), int(lm.y * alto)) for lm in mano_landmarks]

    for idx_a, idx_b in CONEXIONES_MANO:
        cv2.line(frame, coords_px[idx_a], coords_px[idx_b], (0, 255, 120), 2)

    for idx, (cx, cy) in enumerate(coords_px):
        color = (0, 0, 255) if idx == 0 else ((0, 255, 255) if idx in (4, 8, 12, 16, 20) else (255, 100, 0))
        radio = 7 if idx in (0, 4, 8, 12, 16, 20) else 5
        cv2.circle(frame, (cx, cy), radio, color, -1)
        cv2.circle(frame, (cx, cy), radio + 2, (255, 255, 255), 1)
        etiqueta = NOMBRES_PUNTOS_CLAVE.get(idx, str(idx))
        cv2.putText(frame, etiqueta, (cx + 6, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

# --- 3. Modelo LSTM para Secuencias en el Recuadro (ROI) ---
FRAMES_POR_SECUENCIA = 10
TAMANO_ROI = (32, 32)
INPUT_SIZE = TAMANO_ROI[0] * TAMANO_ROI[1] * 3  # 3072 características por frame RGB

class ROILSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=64, num_classes=4):
        super(ROILSTM, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.proj(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

CLASES = ["Hola", "Wena", "ComoEstai", "Nada"]
CLASE_A_IDX = {c: i for i, c in enumerate(CLASES)}

modelo_lstm = ROILSTM(num_classes=len(CLASES))
buffer_frames = deque(maxlen=FRAMES_POR_SECUENCIA)

HAND_TASK_PATH = "modelos/hand_landmarker.task"
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

datos_x = []
etiquetas_y = []
modelo_entrenado = False
ultima_voz = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("\n=== DEMO LSCh ROBUSTA POR RECUADRO (ROI) CON RED LSTM + PUNTOS VISIBLES ===")
print("[H] - Captura secuencia 'HOLA' (Mano dentro del recuadro)")
print("[C] - Captura secuencia 'WENA' (Mano dentro del recuadro)")
print("[K] - Captura secuencia 'COMO ESTAI' (Mano dentro del recuadro)")
print("[N] - Captura secuencia 'NADA' (Recuadro vacío)")
print("[E] - Presiona una vez para ENTRENAR la red LSTM")
print("[ESC] - Salir\n")

while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        continue

    frame = cv2.flip(frame, 1)  # Efecto espejo
    
    y1, y2, x1, x2 = 100, 340, 320, 560

    roi = frame[y1:y2, x1:x2]
    roi_small = cv2.resize(roi, TAMANO_ROI)
    caracteristicas = (roi_small.flatten() / 255.0).astype(np.float32)
    buffer_frames.append(caracteristicas)

    # Dibujar puntos clave de la mano sobre el frame para visualización
    if detector_mano is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultados = detector_mano.detect(mp_image)
        if resultados.hand_landmarks:
            dibujar_puntos_mano(frame, resultados.hand_landmarks[0])
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "PON TU MANO AQUI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    tecla = cv2.waitKey(1) & 0xFF

    # --- Captura de Secuencia Temporal dentro del Recuadro ---
    if len(buffer_frames) == FRAMES_POR_SECUENCIA:
        secuencia_actual = np.array(list(buffer_frames), dtype=np.float32)  # (10, 3072)

        if tecla == ord('h'):
            datos_x.append(secuencia_actual)
            etiquetas_y.append(CLASE_A_IDX["Hola"])
            cv2.putText(frame, f"Guardando secuencia: HOLA ({len(datos_x)})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        elif tecla == ord('c'):
            datos_x.append(secuencia_actual)
            etiquetas_y.append(CLASE_A_IDX["Wena"])
            cv2.putText(frame, f"Guardando secuencia: WENA / PO ({len(datos_x)})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        elif tecla == ord('k'):
            datos_x.append(secuencia_actual)
            etiquetas_y.append(CLASE_A_IDX["ComoEstai"])
            cv2.putText(frame, f"Guardando secuencia: COMO ESTAI ({len(datos_x)})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        elif tecla == ord('n'):
            datos_x.append(secuencia_actual)
            etiquetas_y.append(CLASE_A_IDX["Nada"])
            cv2.putText(frame, f"Guardando secuencia: NADA ({len(datos_x)})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- Entrenamiento de la Red LSTM en Vivo ---
    if tecla == ord('e') and len(datos_x) > 10:
        print(f"Entrenando red LSTM con {len(datos_x)} secuencias...")
        X_tensor = torch.tensor(np.array(datos_x), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(etiquetas_y), dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        modelo_lstm = ROILSTM(num_classes=len(CLASES))
        criterio = nn.CrossEntropyLoss()
        optimizador = torch.optim.Adam(modelo_lstm.parameters(), lr=0.003)

        modelo_lstm.train()
        for epoca in range(40):
            for bx, by in loader:
                optimizador.zero_grad()
                loss = criterio(modelo_lstm(bx), by)
                loss.backward()
                optimizador.step()

        modelo_lstm.eval()
        modelo_entrenado = True
        print(f"¡Modelo LSTM entrenado exitosamente con {len(datos_x)} secuencias!")

    # --- Predicción y Reproducción de Voz ---
    if modelo_entrenado and len(buffer_frames) == FRAMES_POR_SECUENCIA:
        secuencia_actual = np.array(list(buffer_frames), dtype=np.float32)
        sample = torch.tensor(secuencia_actual, dtype=torch.float32).unsqueeze(0)  # (1, 10, 3072)
        with torch.no_grad():
            logits = modelo_lstm(sample)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            prediccion = CLASES[pred_idx]

        cv2.putText(frame, f"Traduccion (LSTM): {prediccion}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        
        if (time.time() - ultima_voz) > 3.0:
            if prediccion == "Hola":
                hablar_async("Hola")
                ultima_voz = time.time()
            elif prediccion == "Wena":
                hablar_async("Wena po")
                ultima_voz = time.time()
            elif prediccion == "ComoEstai":
                hablar_async("¿Cómo estái?")
                ultima_voz = time.time()

    cv2.imshow('Demo LSCh - Tiempo Real (LSTM + Puntos Visibles)', frame)

    if tecla == 27:  # Tecla ESC
        break

if detector_mano is not None:
    detector_mano.close()
cap.release()
cv2.destroyAllWindows()