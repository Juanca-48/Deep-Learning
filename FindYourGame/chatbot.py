# FASE 2 - Descarga, preparación y unión de datos
from huggingface_hub import login
login(token="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")

# Configura KaggleHub para descargar en la carpeta actual
import os
os.environ['KAGGLEHUB_CACHE'] = os.getcwd()
import kagglehub

print("Descargando reviews de Kaggle...")
local_path = kagglehub.dataset_download("andrewmvd/steam-reviews")
print(f"Reviews descargadas en: {local_path}")

# Descarga el dataset de juegos desde HuggingFace
from datasets import load_dataset
import pandas as pd
print("Descargando Steam games desde HuggingFace...")
hf_ds = load_dataset("FronkonGames/steam-games-dataset", split="train")
games = pd.DataFrame(hf_ds)
games.to_csv("steam_games.csv", index=False)
print("Dataset de juegos guardado en steam_games.csv")

# Carga ambos datasets desde archivos locales
reviews = pd.read_csv(os.path.join(local_path, "steam_reviews.csv"))
games = pd.read_csv("steam_games.csv")
print("Juegos columnas:", games.columns.tolist())
print("Reviews columnas:", reviews.columns.tolist())

# Usa 'AppID' y 'app_id' para unir (corrige nombres si necesario)
games.rename(columns={"AppID": "appid"}, inplace=True)
reviews.rename(columns={"app_id": "appid"}, inplace=True)

# Limpieza básica
# Juegos: elimina duplicados y descripciones vacías
games = games.dropna(subset=["appid", "Name"]).drop_duplicates(subset=["appid"])
if "About the game" in games.columns:
    games = games[games["About the game"].astype(str).str.strip() != ""]

# Reviews: elimina reviews vacías
reviews = reviews.dropna(subset=["appid", "review_text"])
reviews["review_text"] = reviews["review_text"].astype(str).str.strip()
reviews = reviews[reviews["review_text"] != ""]

# Unión por appid usando left_on/right_on
merged = pd.merge(
    reviews,
    games,
    left_on="appid",
    right_on="appid",
    how="inner",
    suffixes=("", "_game")
)
print(f"Reviews unidas con metadatos: {merged.shape}")

# Generación de prompts conversacionales para fine-tuning
import random

def shorten(text, max_chars=220):
    return str(text).replace("\n", " ").strip()[:max_chars]

prompts = []
for _, row in merged.iterrows():
    name = row.get("Name", "Juego")
    genre = row.get("Genres", "género desconocido")
    desc = shorten(row.get("About the game", ""), 140)
    review = shorten(row.get("review_text", ""), 180)
    user = random.choice([
        f"¿Me recomiendas un juego de {genre}?",
        f"¿Qué opinan los usuarios de {name}?",
        f"¿Qué juegos similares a {name} existen?"
    ])
    bot = f"Te recomiendo {name}. {desc} Opiniones: '{review}'."
    prompts.append({"text": f"Usuario: {user}\nAsistente: {bot}"})

# Guardar dataset conversacional para fine-tuning
df_prompts = pd.DataFrame(prompts)
df_prompts.to_parquet("train_prompts.parquet", engine="pyarrow")  # O usa engine="fastparquet"

print("Conversational dataset generado en train_prompts.parquet")

# División en entrenamiento y validación
from sklearn.model_selection import train_test_split
train, val = train_test_split(df_prompts, test_size=0.1, random_state=42)
train.to_parquet("train.parquet", engine="pyarrow")
val.to_parquet("val.parquet", engine="pyarrow")
print("train.parquet y val.parquet guardados para fine-tuning")
