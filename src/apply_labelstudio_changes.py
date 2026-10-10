import json

# 📂 fichiers
original_file = "combined_full_data_part1.json"
labelstudio_file = "project-7-at-2026-04-15-10-46-e4bc0df6.json"

# 🔹 1. charger dataset original
with open(original_file, "r", encoding="utf-8") as f:
    original_data = json.load(f)

# 🔹 2. charger export Label Studio
with open(labelstudio_file, "r", encoding="utf-8") as f:
    ls_data = json.load(f)


# 🔥 3. construire mapping text_clean -> action + sentiment
actions = {}

for item in ls_data:
    # structure Label Studio
    text = item["data"]["text_clean"]

    annotations = item.get("annotations", [])
    if not annotations:
        continue

    result = annotations[0]["result"]

    action = None
    sentiment = None

    for r in result:
        if r["from_name"] == "action":
            action = r["value"]["choices"][0]

        if r["from_name"] == "label":
            sentiment = r["value"]["choices"][0]

    actions[text] = {
        "action": action,
        "sentiment": sentiment
    }


# 🔹 4. appliquer modifications
kept_data = []
deleted_data = []

for item in original_data:
    text = item["text_clean"]

    if text in actions:
        info = actions[text]

        if info["action"] == "Supprimer":
            deleted_data.append(item)
            continue

        # garder + update sentiment
        item["sentiment"] = info["sentiment"]

    kept_data.append(item)


# 🔹 5. sauvegarder résultats
with open("data/labeled/final_kept.json", "w", encoding="utf-8") as f:
    json.dump(kept_data, f, ensure_ascii=False, indent=2)

with open("data/labeled/final_deleted.json", "w", encoding="utf-8") as f:
    json.dump(deleted_data, f, ensure_ascii=False, indent=2)


print("✅ Gardés :", len(kept_data))
print("❌ Supprimés :", len(deleted_data))