import cv2
import os
import time

# Nombre de la seña a grabar
CLASE = "como estai"
DURACION_CLIP_SEG = 2.5

RUTA_SALIDA = os.path.join("dataset_lsch", CLASE)
os.makedirs(RUTA_SALIDA, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 30.0

print(f"\n=== GRABADOR SEGURO DE CLIPS PARA '{CLASE.upper()}' ===")
print("Presiona [ESPACIO] para iniciar la grabación.")
print("Presiona [ESC] para salir.\n")

clip_count = len([f for f in os.listdir(RUTA_SALIDA) if f.endswith('.mp4')])

while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        continue

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Clase: {CLASE} | Clips listos: {clip_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Presiona ESPACIO para grabar", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Grabador LSCh", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord(' '):
        # Cuenta regresiva de 3 segundos
        for i in range(3, 0, -1):
            ret, temp_frame = cap.read()
            if not ret:
                break
            temp_frame = cv2.flip(temp_frame, 1)
            cv2.putText(temp_frame, f"PREPARATE: {i}", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow("Grabador LSCh", temp_frame)
            cv2.waitKey(1000)

        # Configurar grabador con códec 'mp4v'
        nombre_archivo = os.path.join(RUTA_SALIDA, f"{CLASE}_{clip_count + 1:02d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(nombre_archivo, fourcc, fps, (ancho, alto))

        tiempo_inicio = time.time()
        print(f"Grabando {nombre_archivo}...")

        # Grabar estrictamente durante los segundos configurados
        while (time.time() - tiempo_inicio) < DURACION_CLIP_SEG:
            ret, rec_frame = cap.read()
            if not ret:
                break
            rec_frame = cv2.flip(rec_frame, 1)
            out.write(rec_frame)

            cv2.putText(rec_frame, "● GRABANDO...", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Grabador LSCh", rec_frame)
            cv2.waitKey(1)

        # CIERRE OBLIGATORIO DEL ARCHIVO (Genera el moov atom)
        out.release()
        clip_count += 1
        print("¡Video guardado y cerrado correctamente!")
        time.sleep(0.5)

    elif tecla == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()