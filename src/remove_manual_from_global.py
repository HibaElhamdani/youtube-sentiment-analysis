import json

# Paths

global_file = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\data\processed\comments_clean.json"
manual_me_file = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\manual_500_hibalab.json"
manual_friend_file = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\manual_500_friend.json"

output_file = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\data\processed\comments_for_llm.json"


# Load files
with open(global_file, "r", encoding="utf-8") as f:
    global_data = json.load(f)

with open(manual_me_file, "r", encoding="utf-8") as f:
    manual_me = json.load(f)

with open(manual_friend_file, "r", encoding="utf-8") as f:
    manual_friend = json.load(f)

# Collect all manual comment_ids
manual_ids = {item["comment_id"] for item in manual_me}
manual_ids.update(item["comment_id"] for item in manual_friend)

# Remove manual comments from global dataset
filtered_data = [item for item in global_data if item["comment_id"] not in manual_ids]

# Save cleaned dataset for LLM
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print("✅ Dataset prêt pour LLM.")
print("Total original:", len(global_data))
print("Total manual removed:", len(manual_ids))
print("Total remaining for LLM:", len(filtered_data))