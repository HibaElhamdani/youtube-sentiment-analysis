import json

# 📂 fichiers
file_g1 = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\data\labeled\llm_datasetG1.json"
file_g2 = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\data\labeled\llm_datasetG2.json"
output_file = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\data\labeled\llm_dataset_final.json"

# 🔹 charger fichiers
with open(file_g1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(file_g2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

# 🔥 dictionnaire pour supprimer doublons
merged = {}

# 🔹 ajouter dataset 1
for item in data1:
    cid = item.get("comment_id")
    if cid:
        merged[cid] = item

# 🔹 ajouter dataset 2 (écrase si même id → priorité G2)
for item in data2:
    cid = item.get("comment_id")
    if cid:
        merged[cid] = item

# 🔹 résultat final
final_data = list(merged.values())

# 💾 sauvegarde
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print("✅ Fusion terminée sans doublons")
print("📊 Nombre total :", len(final_data))