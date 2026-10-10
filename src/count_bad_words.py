#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_bad_words.py
==================
Script pour:
1. Compter les gros mots en Darija (MOTS ENTIERS UNIQUEMENT)
2. SUPPRIMER les commentaires avec gros mots
3. Sauvegarder les commentaires supprimés dans un fichier séparé

Usage:
    python src/clean_bad_words.py
"""

import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime

# ======================================================
# CONFIGURATION
# ======================================================

BASE_DIR = Path(r"C:\Users\elham\Desktop\youtube-sentiment-analysis")

FILES_TO_CLEAN = [
    {
        "input": BASE_DIR / "data" / "processed" / "comments_for_llm.json",
        "output_clean": BASE_DIR / "data" / "processed" / "comments_for_llm_clean.json",
        "output_removed": BASE_DIR / "data" / "processed" / "comments_removed_bad_words.json",
    },
]

# ======================================================
# LISTE DES GROS MOTS (DARIJA + ARABIZI)
# ======================================================

BAD_WORDS = [
    # Groupe 1: قلوة et variantes
    "9alwa", "9lawi", "9lwa", "9liwa", "9alwi",
    "قلوة", "قلاوي", "قليوة", "قلواني",
    
    # Groupe 2: زب et variantes
    "زب", "زبي", "زبوبا",
    "zbi", "zeb", "zboba", "zabi", "zab", "zebbi",
    
    # Groupe 3: حوا et variantes
    "hwaya", "l7wa", "7wa", "7waya",
    "حوا", "حواك", "حواتك", "حواها",
    
    # Groupe 4: قحبة et variantes
    "قحبة", "قحاب", "9ahba", "97ab", "9a7ba","تقحبين", "كرو", "قلوى ","kro"," kri","kr" 
    
    # Groupe 5: زامل et variantes
    "زوامل", "زامل", "زملتو",
    "zamel", "zwamel", "zamil",
    
    # Groupe 6: Autres
    "طبون", "taboun", "tboun", "tbon","karoo","kar", "كارو","كاو","tkfat","تكفات"
]

# ======================================================
# FONCTIONS
# ======================================================

def load_json(filepath):
    """Charge un fichier JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath):
    """Sauvegarde en JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    size_kb = filepath.stat().st_size / 1024
    print(f"   💾 Sauvegardé: {filepath.name} ({len(data)} items, {size_kb:.1f} KB)")


def find_bad_words_whole(text, bad_words_list):
    """
    Trouve les gros mots ENTIERS uniquement (pas les sous-chaînes).
    
    Utilise des frontières de mots (word boundaries) pour s'assurer
    que le mot est séparé et non partie d'un autre mot.
    """
    if not text:
        return []
    
    found = []
    text_lower = text.lower()
    
    for word in bad_words_list:
        word_lower = word.lower()
        
        # Créer un pattern regex avec frontières de mots
        # \b = word boundary (frontière de mot)
        # re.escape() pour échapper les caractères spéciaux
        
        # Pour les mots arabes, on utilise une approche différente
        # car \b ne fonctionne pas bien avec l'arabe
        
        if any('\u0600' <= c <= '\u06FF' for c in word):
            # Mot arabe: chercher avec espaces/ponctuation autour
            # Pattern: début ou espace/ponctuation AVANT + mot + fin ou espace/ponctuation APRÈS
            pattern = r'(?:^|[\s\.,!?،؟\-_])' + re.escape(word_lower) + r'(?:$|[\s\.,!?،؟\-_])'
        else:
            # Mot latin/arabizi: utiliser \b (word boundary)
            pattern = r'\b' + re.escape(word_lower) + r'\b'
        
        if re.search(pattern, text_lower):
            found.append(word)
    
    return found


def clean_file(config, bad_words_list):
    """Nettoie un fichier en supprimant les commentaires avec gros mots ENTIERS."""
    
    input_path = config["input"]
    output_clean_path = config["output_clean"]
    output_removed_path = config["output_removed"]
    
    print(f"\n{'='*60}")
    print(f"📂 Fichier: {input_path.name}")
    print(f"{'='*60}")
    
    if not input_path.exists():
        print(f"❌ Fichier non trouvé!")
        return None
    
    # Charger les données
    data = load_json(input_path)
    print(f"📊 Total commentaires: {len(data)}")
    
    # Séparer les commentaires
    clean_comments = []
    removed_comments = []
    word_counter = Counter()
    
    for item in data:
        text = item.get("text_clean", "") or item.get("text", "")
        
        # ✅ Utiliser la nouvelle fonction qui cherche les MOTS ENTIERS
        found_words = find_bad_words_whole(text, bad_words_list)
        
        if found_words:
            # Commentaire avec gros mots → à supprimer
            item_with_info = item.copy()
            item_with_info["bad_words_found"] = found_words
            item_with_info["removed_at"] = datetime.now().isoformat()
            removed_comments.append(item_with_info)
            
            for word in found_words:
                word_counter[word] += 1
        else:
            # Commentaire propre → à garder
            clean_comments.append(item)
    
    # Afficher statistiques
    print(f"\n📊 RÉSULTATS:")
    print(f"   ✅ Commentaires propres: {len(clean_comments)}")
    print(f"   ❌ Commentaires supprimés: {len(removed_comments)}")
    
    pct_removed = 0
    pct_clean = 0
    if len(data) > 0:
        pct_removed = len(removed_comments) / len(data) * 100
        pct_clean = len(clean_comments) / len(data) * 100
        print(f"   📈 Taux de suppression: {pct_removed:.2f}%")
        print(f"   📈 Taux de conservation: {pct_clean:.2f}%")
    
    # Afficher fréquence des gros mots
    if word_counter:
        print(f"\n📋 Gros mots trouvés (par fréquence):")
        print("-" * 40)
        for word, count in word_counter.most_common(15):
            print(f"   • '{word}': {count} fois")
    
    # Afficher quelques exemples supprimés
    if removed_comments:
        print(f"\n📝 Exemples supprimés ({min(5, len(removed_comments))} premiers):")
        print("-" * 40)
        for i, item in enumerate(removed_comments[:5], 1):
            text = item['text_clean'][:50] + "..." if len(item['text_clean']) > 50 else item['text_clean']
            print(f"   [{i}] \"{text}\"")
            print(f"       Mots: {item['bad_words_found']}")
    
    # Sauvegarder les fichiers
    print(f"\n💾 SAUVEGARDE:")
    save_json(clean_comments, output_clean_path)
    save_json(removed_comments, output_removed_path)
    
    return {
        "input_file": str(input_path),
        "total_original": len(data),
        "total_clean": len(clean_comments),
        "total_removed": len(removed_comments),
        "percentage_removed": pct_removed,
        "word_frequencies": dict(word_counter),
    }


def test_detection():
    """Test pour vérifier que la détection fonctionne correctement."""
    
    print("\n" + "=" * 60)
    print("🧪 TEST DE DÉTECTION (mots entiers vs sous-mots)")
    print("=" * 60)
    
    test_cases = [
        # (texte, doit_être_détecté)
        ("zbi", True),                    # ✅ mot seul
        ("wa zbi nta", True),             # ✅ mot séparé par espaces
        ("hada zbi!", True),              # ✅ mot avec ponctuation
        ("tzbi3", False),                 # ❌ sous-mot (NE PAS détecter)
        ("azbin", False),                 # ❌ sous-mot
        ("9ahba", True),                  # ✅ mot seul
        ("ya 9ahba", True),               # ✅ mot séparé
        ("m9ahba", False),                # ❌ sous-mot
        ("زب", True),                     # ✅ mot arabe seul
        ("يا زب", True),                  # ✅ mot arabe séparé
        ("قحبة", True),                   # ✅ mot arabe seul
        ("هادي قحبة نتاع", True),         # ✅ mot arabe séparé
    ]
    
    print("\n{:<30} {:<15} {:<15} {:<10}".format("Texte", "Attendu", "Résultat", "Status"))
    print("-" * 70)
    
    all_passed = True
    for text, should_detect in test_cases:
        found = find_bad_words_whole(text, BAD_WORDS)
        detected = len(found) > 0
        
        status = "✅" if detected == should_detect else "❌"
        if detected != should_detect:
            all_passed = False
        
        expected = "DÉTECTER" if should_detect else "IGNORER"
        result = "DÉTECTÉ" if detected else "IGNORÉ"
        
        print(f"{text:<30} {expected:<15} {result:<15} {status}")
    
    print("-" * 70)
    if all_passed:
        print("✅ Tous les tests passés!")
    else:
        print("⚠️ Certains tests ont échoué")
    
    return all_passed


def main():
    print("\n" + "=" * 60)
    print("🧹 NETTOYAGE DES GROS MOTS EN DARIJA")
    print("   (Détection de MOTS ENTIERS uniquement)")
    print("=" * 60)
    
    # Lancer le test d'abord
    test_detection()
    
    print(f"\n📋 Liste des mots à filtrer ({len(BAD_WORDS)} mots):")
    print(f"   {BAD_WORDS[:8]}...")
    
    all_results = []
    
    for config in FILES_TO_CLEAN:
        result = clean_file(config, BAD_WORDS)
        if result:
            all_results.append(result)
    
    # Résumé global
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ GLOBAL")
    print("=" * 60)
    
    total_original = sum(r["total_original"] for r in all_results)
    total_clean = sum(r["total_clean"] for r in all_results)
    total_removed = sum(r["total_removed"] for r in all_results)
    
    print(f"\n   📥 Total original: {total_original}")
    print(f"   ✅ Total propre: {total_clean}")
    print(f"   ❌ Total supprimé: {total_removed}")
    
    if total_original > 0:
        print(f"   📈 Taux global de suppression: {total_removed/total_original*100:.2f}%")
    
    # Sauvegarder le rapport
    report_path = BASE_DIR / "data" / "cleaning_report.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "detection_mode": "WHOLE_WORDS_ONLY",
        "bad_words_list": BAD_WORDS,
        "summary": {
            "total_original": total_original,
            "total_clean": total_clean,
            "total_removed": total_removed,
        },
        "details": all_results
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 Rapport: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ NETTOYAGE TERMINÉ!")
    print("=" * 60)
    
    print(f"\n📁 Fichiers générés:")
    for config in FILES_TO_CLEAN:
        print(f"   • {config['output_clean'].name} (commentaires propres)")
        print(f"   • {config['output_removed'].name} (commentaires supprimés)")


if __name__ == "__main__":
    main()