#Toutes les constantes sont ici
#Importe le module os qui permet de lire les variables d'environnement du système
import os
from dotenv import load_dotenv

load_dotenv()

# API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


# Chemins
RAW_DATA_PATH = "data/raw/comments.json"
PROCESSED_DATA_PATH = "data/processed/comments_clean.json"
LABELED_DATA_PATH = "data/labeled/dataset_final.json"
MODELS_PATH = "models_saved/"
FIGURES_PATH = "reports/figures/"

# Paramètres
MAX_COMMENTS = 60000
LABELS = ["positif", "négatif", "neutre"]
MLFLOW_EXPERIMENT_NAME = "youtube-sentiment-analysis"