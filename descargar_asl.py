import os
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

REPO_ID = "akasheroor/American-Sign-Language-Dataset"
OUTPUT_DIR = "data/asl/videos"
CSV_INDEX = "data/asl/Aslense_Dataset.csv"
CANDIDATOS_CSV = "data/asl/candidatos_asl.csv"

# Palabras en inglés del dataset ASL:
# - Equivalentes para reforzar señas actuales (hello, cool, fine, how, etc.)
# - 7 NUEVAS palabras cuyos gestos tienen overlapping directo entre ASL y LSCh (español chileno):
#   1. bye    -> "chao" (saludo de despedida con la mano)
#   2. yes    -> "si" (puño asintiendo arriba/abajo)
#   3. no     -> "no" (índice y medio cerrando con pulgar)
#   4. good   -> "gracias" / "bacan" (mano desde el mentón hacia adelante)
#   5. eat    -> "comer" (mano cerrada llevada a la boca)
#   6. drink  -> "tomar" (mano en C inclinándose hacia la boca)
#   7. please -> "porfa" (mano abierta frotando el pecho)
#   8. stop   -> "para" (canto de la mano sobre la palma)
PALABRAS_OBJETIVO = [
    "hello", "cool", "welcome", "friend", "how", "you", "fine", "what",
    "bye", "yes", "no", "good", "eat", "drink", "please", "stop"
]
VIDEOS_POR_PALABRA = 10  # 10 videos por palabra

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Actualizando lista de candidatos desde Aslense_Dataset.csv...")
if not os.path.exists(CSV_INDEX):
    p = hf_hub_download(REPO_ID, "Aslense Dataset.csv", repo_type="dataset")
    shutil.copy(p, CSV_INDEX)

df = pd.read_csv(CSV_INDEX)
sub = df[df["word"].str.lower().str.strip().isin(PALABRAS_OBJETIVO)].copy()

api = HfApi()
files = api.list_repo_files(REPO_ID, repo_type="dataset")
mapa = {f.split("/")[-1]: f for f in files if f.endswith(".mp4")}
sub["repo_path"] = sub["videos"].map(mapa)
sub = sub.dropna(subset=["repo_path"])
sub.to_csv(CANDIDATOS_CSV, index=False)

seleccion = sub.groupby("word").head(VIDEOS_POR_PALABRA)
print(f"Descargando {len(seleccion)} videos de ASL en paralelo para las {len(PALABRAS_OBJETIVO)} palabras...")

def descargar_uno(row):
    palabra = str(row["word"]).lower().strip()
    carpeta_palabra = os.path.join(OUTPUT_DIR, palabra)
    os.makedirs(carpeta_palabra, exist_ok=True)
    destino = os.path.join(carpeta_palabra, os.path.basename(row["repo_path"]))

    if not os.path.exists(destino):
        try:
            cache_path = hf_hub_download(REPO_ID, row["repo_path"], repo_type="dataset")
            shutil.copy(cache_path, destino)
        except Exception as e:
            print(f"Error descargando {row['repo_path']}: {e}")
            return None

    return {"word": palabra, "video_name": row["videos"], "local_path": destino}

rutas_locales = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futuros = [executor.submit(descargar_uno, row) for _, row in seleccion.iterrows()]
    for fut in as_completed(futuros):
        res = fut.result()
        if res is not None:
            rutas_locales.append(res)

df_local = pd.DataFrame(rutas_locales)
df_local.to_csv("data/asl/videos_descargados.csv", index=False)
print(f"¡Listo! {len(df_local)} videos guardados en '{OUTPUT_DIR}' y registrados en 'data/asl/videos_descargados.csv'.")


