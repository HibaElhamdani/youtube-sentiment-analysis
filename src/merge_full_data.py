import json

# =========================
# 📂 FILES
# =========================
file_raw = "data/processed/comments_for_llm.json"
file_labeled = "data/labeled/final_dataset_clean_labelise_verifie.json"


# =========================
# 🔹 1. LOAD RAW DATA (full info)
# =========================
with open(file_raw, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# text_clean -> full metadata
text_to_full = {}

for item in raw_data:
    text = item["text_clean"].strip()
    text_to_full[text] = item

print("✔ Raw mapping :", len(text_to_full))


# =========================
# 🔹 2. LOAD LABELED DATA
# =========================
with open(file_labeled, "r", encoding="utf-8") as f:
    labeled_data = json.load(f)


# =========================
# 🔥 3. MERGE
# =========================
final_data = []
missing = 0

for item in labeled_data:

    text = item["text_clean"].strip()

    full_item = text_to_full.get(text)

    if full_item is None:
        missing += 1
        continue

    merged = {
        "comment_id": full_item.get("comment_id"),
        "text": full_item.get("text"),
        "text_clean": full_item.get("text_clean"),
        "tokens": full_item.get("tokens"),
        "date": full_item.get("date"),
        "likes": full_item.get("likes"),
        "channel": full_item.get("channel"),
        "sentiment": item["sentiment"]
    }

    final_data.append(merged)


# =========================
# 🔹 4. SAVE FINAL DATASET
# =========================
output_file = "data/labeled/final_dataset.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)


# =========================
# 🔥 STATS
# =========================
print("\n========== RESULT ==========")
print("✔ Final size :", len(final_data))
print("❌ Missing matches :", missing)
print("💾 Saved to :", output_file)