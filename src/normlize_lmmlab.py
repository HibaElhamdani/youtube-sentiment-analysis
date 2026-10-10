import json
import unicodedata
import re

# ═══════════════════════════════════════
# NORMALISATION SENTIMENT
# ═══════════════════════════════════════

def normalize_sentiment(label):
    if not label:
        return None

    # minuscule + strip
    label = label.lower().strip()

    # enlever accents (é → e)
    label = unicodedata.normalize("NFKD", label)
    label = "".join(c for c in label if not unicodedata.combining(c))

    # enlever caractères spéciaux
    label = re.sub(r"[^a-z]", "", label)

    # mapping final
    if "positif" in label:
        return "positif"
    elif "negatif" in label:
        return "negatif"
    elif "neutre" in label:
        return "neutre"
    else:
        return label


# ═══════════════════════════════════════
# CHARGER FICHIER
# ═══════════════════════════════════════

FILE_PATH = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\data\labeled\llm_dataset_final.json"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ═══════════════════════════════════════
# MODIFICATION DIRECTE
# ═══════════════════════════════════════

for item in data:
    if "sentiment" in item:
        item["sentiment"] = normalize_sentiment(item["sentiment"])

# ═══════════════════════════════════════
# ÉCRASER LE FICHIER ORIGINAL
# ═══════════════════════════════════════

with open(FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✔ Fichier modifié directement avec succès")