import cv2
import numpy as np
import os
import json
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATASET_PATH = "dataset_lsch"
REPORTE_OVERLAP_PATH = "data/asl/reporte_overlapping.json"
MODELO_PATH = "modelos/modelo_lsch_mediapipe_lstm.pth"
HAND_TASK_PATH = "modelos/hand_landmarker.task"
FRAMES_POR_SECUENCIA = 10
NUM_FEATURES = 42  # 21 puntos * (x, y) normalizados respecto a la muñeca

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

def crear_detector_mano():
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
    """
    Extrae 42 características híbridas para cada frame:
    - 40 coordenadas (x, y) de las 20 articulaciones relativas a la muñeca (escaladas por tamaño de mano),
      para aprender la FORMA exacta de los dedos sin importar la distancia a la cámara.
    - 2 coordenadas globales (x, y) de la muñeca (punto 0), para que la LSTM sí distinga la TRAYECTORIA
      y si la mano se mueve hacia la boca (comer/tomar), pecho (porfa) o de lado a lado (hola/chao).
    """
    coords = np.array([[lm.x, lm.y] for lm in mano_landmarks], dtype=np.float32)  # (21, 2)
    muneca = coords[0].copy()  # (2,)
    relativos = coords[1:] - muneca  # (20, 2)
    max_dist = np.max(np.linalg.norm(relativos, axis=1))
    if max_dist > 1e-5:
        relativos = relativos / max_dist
    return np.concatenate([muneca, relativos.flatten()]).astype(np.float32)  # (42,)

def extraer_ventanas_overlapping(ruta_video, detector, num_frames=FRAMES_POR_SECUENCIA, stride=4, es_clase_nada=False):
    cap = cv2.VideoCapture(ruta_video)
    secuencia_puntos = []
    
    while cap.isOpened():
        exito, frame = cap.read()
        if not exito:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultados = detector.detect(mp_image)
        
        puntos_frame = np.zeros(NUM_FEATURES, dtype=np.float32)
        if resultados.hand_landmarks:
            puntos_frame = normalizar_puntos_mano(resultados.hand_landmarks[0])
            
        secuencia_puntos.append(puntos_frame)
            
    cap.release()
    
    if len(secuencia_puntos) < 3:
        return []
        
    if es_clase_nada:
        indices_global = np.linspace(0, len(secuencia_puntos) - 1, num_frames, dtype=int)
        return [np.array([secuencia_puntos[i] for i in indices_global], dtype=np.float32)]

    # Filtrar SOLO los frames donde sí se detectó la mano para evitar secuencias llenas de ceros
    frames_con_mano = [p for p in secuencia_puntos if np.sum(np.abs(p)) > 0]
    if len(frames_con_mano) < 5:
        return []

    ventanas = []
    # 1. Muestra uniforme de todos los frames con mano activa (inicio a fin del gesto)
    indices_mano = np.linspace(0, len(frames_con_mano) - 1, num_frames, dtype=int)
    ventanas.append(np.array([frames_con_mano[i] for i in indices_mano], dtype=np.float32))

    # 2. Ventanas deslizantes con overlapping temporal sobre los frames con mano activa
    if len(frames_con_mano) >= num_frames:
        pasos = np.linspace(0, len(frames_con_mano) - num_frames, min(3, len(frames_con_mano) - num_frames + 1), dtype=int)
        for inicio in np.unique(pasos):
            sub = np.array(frames_con_mano[inicio:inicio + num_frames], dtype=np.float32)
            ventanas.append(sub)

    return ventanas

def aumentar_secuencia(secuencia, num_variaciones=5):
    variaciones = [secuencia]
    es_vacia = np.sum(np.abs(secuencia)) == 0
    for _ in range(num_variaciones):
        if es_vacia:
            variaciones.append(secuencia.copy())
        else:
            ruido = np.random.normal(0, 0.015, secuencia.shape).astype(np.float32)
            escala = np.random.uniform(0.94, 1.06)
            mascara = (secuencia != 0).astype(np.float32)
            nueva = (secuencia * escala + ruido) * mascara
            variaciones.append(nueva.astype(np.float32))
    return variaciones

if __name__ == "__main__":
    print("=== ENTRENANDO RED NEURONAL LSTM CON 11 CLASES EN CHILENO (LSCh + 7 NUEVAS DESDE ASL) ===")
    X, y = [], []
    
    # Reunir clases base de dataset_lsch + las nuevas clases chilenas aprobadas en el reporte de overlapping
    clases_set = set([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    reporte = {}
    if os.path.exists(REPORTE_OVERLAP_PATH):
        with open(REPORTE_OVERLAP_PATH, "r", encoding="utf-8") as f:
            reporte = json.load(f)
        for v in reporte.get("videos", []):
            if v.get("aprobado_para_entrenar"):
                clases_set.add(v["clase_lsch_asignada"])

    clases = sorted(list(clases_set))
    clase_a_idx = {c: i for i, c in enumerate(clases)}
    print(f"Total de clases en español chileno ({len(clases)}): {clases}")
    
    detector = crear_detector_mano()
    
    # 1. Cargar videos propios en español (dataset_lsch: hola, wena po, como estai, nada)
    for etiqueta in sorted(os.listdir(DATASET_PATH)):
        carpeta = os.path.join(DATASET_PATH, etiqueta)
        if not os.path.isdir(carpeta) or etiqueta not in clase_a_idx:
            continue
        idx_clase = clase_a_idx[etiqueta]
        archivos = [f for f in os.listdir(carpeta) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"[LSCh Local] Procesando {len(archivos)} videos para '{etiqueta}'...")
        
        for archivo in archivos:
            ruta = os.path.join(carpeta, archivo)
            es_nada = (etiqueta == "nada")
            ventanas = extraer_ventanas_overlapping(ruta, detector, FRAMES_POR_SECUENCIA, stride=4, es_clase_nada=es_nada)
            for secuencia in ventanas:
                for seq_aug in aumentar_secuencia(secuencia, num_variaciones=5):
                    X.append(seq_aug)
                    y.append(idx_clase)

    # 2. Cargar videos en inglés (ASL) aprobados por overlapping (SOLO las 7 nuevas palabras chilenas para no distorsionar las 3 señas propias)
    asl_integrados = 0
    conteo_por_nueva_clase = {}
    if reporte:
        aprobados = [v for v in reporte.get("videos", []) if v.get("aprobado_para_entrenar") and v.get("tipo") == "nueva_clase_chilena"]
        print(f"\n[Dataset Inglés ASL -> Chileno] Procesando {len(aprobados)} videos de las 7 nuevas palabras chilenas...")
        for item in aprobados:
            clase_destino = item["clase_lsch_asignada"]
            ruta_asl = item["ruta"]
            if clase_destino in clase_a_idx and os.path.exists(ruta_asl):
                idx_c = clase_a_idx[clase_destino]
                ventanas = extraer_ventanas_overlapping(ruta_asl, detector, FRAMES_POR_SECUENCIA, stride=4, es_clase_nada=False)
                for secuencia in ventanas[:3]:
                    for seq_aug in aumentar_secuencia(secuencia, num_variaciones=3):
                        X.append(seq_aug)
                        y.append(idx_c)
                asl_integrados += 1
                conteo_por_nueva_clase[clase_destino] = conteo_por_nueva_clase.get(clase_destino, 0) + 1

        print("Videos de ASL integrados por palabra chilena:")
        for c_nom, cant in sorted(conteo_por_nueva_clase.items()):
            print(f"  -> '{c_nom}': {cant} videos en inglés convertidos a secuencias LSTM")

    detector.close()

    if len(X) > 0:
        print(f"\nTotal de secuencias temporales 3D entrenadas: {len(X)}")
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)  # (N, 10, 42)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        modelo = SignLanguageLSTM(
            input_size=NUM_FEATURES,
            hidden_size=64,
            num_layers=2,
            num_classes=len(clases)
        )

        criterio = nn.CrossEntropyLoss()
        optimizador = torch.optim.Adam(modelo.parameters(), lr=0.003)

        epocas = 120
        modelo.train()
        for epoca in range(1, epocas + 1):
            perdida_total = 0.0
            correctos = 0
            total = 0
            for batch_x, batch_y in loader:
                optimizador.zero_grad()
                salidas = modelo(batch_x)
                loss = criterio(salidas, batch_y)
                loss.backward()
                optimizador.step()

                perdida_total += loss.item() * batch_x.size(0)
                _, preds = torch.max(salidas, 1)
                correctos += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            if epoca % 20 == 0 or epoca == 1:
                acc = (correctos / total) * 100
                print(f"Época [{epoca}/{epocas}] - Pérdida: {perdida_total/total:.4f} - Precisión: {acc:.1f}%")

        os.makedirs("modelos", exist_ok=True)
        checkpoint = {
            "model_state_dict": modelo.state_dict(),
            "clases": clases,
            "input_size": NUM_FEATURES,
            "hidden_size": 64,
            "num_layers": 2,
            "frames_por_secuencia": FRAMES_POR_SECUENCIA,
            "normalizado_muneca": True,
            "asl_videos_integrados": asl_integrados
        }
        torch.save(checkpoint, MODELO_PATH)
        print(f"\n¡MODELO LSTM DE {len(clases)} CLASES CHILENAS GUARDADO EN '{MODELO_PATH}'!")
