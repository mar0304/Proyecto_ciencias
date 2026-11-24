import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATASET_PATH = "dataset_lsch"
MODELO_PATH = "modelos/modelo_lsch_videos_lstm.pth"
FRAMES_POR_SECUENCIA = 10
TAMANO_FRAME = (32, 32)
NUM_FEATURES = TAMANO_FRAME[0] * TAMANO_FRAME[1]  # 1024 características por frame

# --- Definición de la Red Neuronal LSTM para Secuencias de Video ---
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
        # x shape: (batch_size, seq_len, 1024)
        x = self.proj(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def procesar_video(ruta_video, num_frames=10):
    cap = cv2.VideoCapture(ruta_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        cap.release()
        return None
        
    step = max(1, total_frames // num_frames)
    secuencia_frames = []
    
    count = 0
    while cap.isOpened() and len(secuencia_frames) < num_frames:
        exito, frame = cap.read()
        if not exito:
            break
            
        if count % step == 0:
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resalta = cv2.resize(gris, TAMANO_FRAME)
            # Normalización estándar entre 0.0 y 1.0
            vector_norm = (resalta.flatten() / 255.0).astype(np.float32)
            secuencia_frames.append(vector_norm)
            
        count += 1
        
    cap.release()
    
    if len(secuencia_frames) == num_frames:
        # Retorna matriz 2D de forma (num_frames, 1024) para la LSTM
        return np.array(secuencia_frames, dtype=np.float32)
    return None

def aumentar_secuencia_video(secuencia, num_variaciones=5):
    variaciones = [secuencia]
    for _ in range(num_variaciones):
        ruido = np.random.normal(0, 0.02, secuencia.shape).astype(np.float32)
        brillo = np.random.uniform(0.92, 1.08)
        nueva = np.clip(secuencia * brillo + ruido, 0.0, 1.0)
        variaciones.append(nueva.astype(np.float32))
    return variaciones

def cargar_dataset():
    X = []
    y = []
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: La carpeta '{DATASET_PATH}' no existe.")
        return None, None, None

    clases = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    print(f"Clases detectadas: {clases}")
    
    for idx_clase, etiqueta in enumerate(clases):
        carpeta_clase = os.path.join(DATASET_PATH, etiqueta)
        archivos = [f for f in os.listdir(carpeta_clase) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"Procesando {len(archivos)} videos para '{etiqueta}'...")
        
        for archivo in archivos:
            ruta = os.path.join(carpeta_clase, archivo)
            matriz_secuencia = procesar_video(ruta, FRAMES_POR_SECUENCIA)
            
            if matriz_secuencia is not None:
                for seq_aug in aumentar_secuencia_video(matriz_secuencia):
                    X.append(seq_aug)
                    y.append(idx_clase)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), clases

if __name__ == "__main__":
    print("=== ENTRENANDO MODELO LSTM DESDE VIDEOS LSCh ===")
    X, y, clases = cargar_dataset()
    
    if X is not None and len(X) > 0:
        X_tensor = torch.tensor(X, dtype=torch.float32)  # (N, 10, 1024)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        modelo = VideoSequenceLSTM(
            input_size=NUM_FEATURES,
            hidden_size=128,
            num_layers=2,
            num_classes=len(clases)
        )

        criterio = nn.CrossEntropyLoss()
        optimizador = torch.optim.Adam(modelo.parameters(), lr=0.002)

        epocas = 80
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

            if epoca % 10 == 0 or epoca == 1:
                acc = (correctos / total) * 100
                print(f"Época [{epoca}/{epocas}] - Pérdida: {perdida_total/total:.4f} - Precisión: {acc:.1f}%")

        os.makedirs("modelos", exist_ok=True)
        checkpoint = {
            "model_state_dict": modelo.state_dict(),
            "clases": clases,
            "input_size": NUM_FEATURES,
            "hidden_size": 128,
            "num_layers": 2,
            "frames_por_secuencia": FRAMES_POR_SECUENCIA,
            "tamano_frame": TAMANO_FRAME
        }
        torch.save(checkpoint, MODELO_PATH)
        print(f"\n¡Modelo LSTM entrenado y guardado exitosamente en '{MODELO_PATH}'!")