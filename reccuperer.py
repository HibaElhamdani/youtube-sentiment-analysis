import json
import pandas as pd

def load_labeled_data(export_file_path):
    """
    Charger les données labellisées depuis le fichier JSON exporté de Label Studio
    """
    with open(export_file_path, 'r', encoding='utf-8') as f:
        labeled_data = json.load(f)
    
    processed_data = []
    
    for item in labeled_data:
        # Données originales
        original_data = item.get('data', {})
        
        # Annotations (labels)
        annotations = item.get('annotations', [])
        
        # Extraire le sentiment
        sentiment = None
        if annotations:
            latest_annotation = annotations[-1]
            results = latest_annotation.get('result', [])
            
            for result in results:
                if result.get('type') == 'choices':
                    choices = result.get('value', {}).get('choices', [])
                    if choices:
                        sentiment = choices[0]  # positif, negatif, neutre
        
        # Fusionner les données
        row = {
            'comment_id': original_data.get('comment_id'),
            'text': original_data.get('text'),
            'text_clean': original_data.get('text_clean'),
            'tokens': original_data.get('tokens'),
            'date': original_data.get('date'),
            'likes': original_data.get('likes'),
            'channel': original_data.get('channel'),
            'lang_hint': original_data.get('lang_hint'),
            'sentiment': sentiment  # ✅ Nouvelle colonne
        }
        processed_data.append(row)
    
    return pd.DataFrame(processed_data)


# ========================================
# UTILISATION
# ========================================

# 1. Charger le fichier exporté
df = load_labeled_data('C:\\Users\\elham\\Desktop\\youtube-sentiment-analysis\\h.json')

# 2. Afficher les résultats
print("📊 Données labellisées :")
print(df[['text_clean', 'sentiment']].head(10))

# 3. Statistiques
print("\n📈 Distribution des sentiments :")
print(df['sentiment'].value_counts())

# 4. Sauvegarder
df.to_json('data\\labeled\\labeled_comments1.json', orient='records', force_ascii=False, indent=2)

print("\n✅ Données sauvegardées !")