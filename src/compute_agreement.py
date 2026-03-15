import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Files
my_file = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\data\labeled\labeled_comments100.json"
partner_file = r"C:\Users\elham\Desktop\youtube-sentiment-analysis\data\labeled\friend_overlap_100.json"

# Load
with open(my_file, "r", encoding="utf-8") as f:
    my_data = json.load(f)

with open(partner_file, "r", encoding="utf-8") as f:
    partner_data = json.load(f)

# Convert to DataFrame
df1 = pd.DataFrame(my_data)
df2 = pd.DataFrame(partner_data)

# Merge on comment_id
merged = df1.merge(df2, on="comment_id", suffixes=("_me", "_partner"))

print("Nombre de commentaires comparés:", len(merged))

labels1 = merged["sentiment_me"]
labels2 = merged["sentiment_partner"]

# Simple agreement
agreement = (labels1 == labels2).mean()

# Cohen’s Kappa
kappa = cohen_kappa_score(labels1, labels2)

print("Simple Agreement:", round(agreement * 100, 2), "%")
print("Cohen’s Kappa:", round(kappa, 3))
disagreements = merged[labels1 != labels2]
print(disagreements[["text_me", "text_partner", "sentiment_me", "sentiment_partner"]])