#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_llm_data.py
===================
Script pour minimiser les données JSON avant envoi au LLM.

Deux cas:
1. comments_for_llm_clean.json → garder: comment_id, text_clean
2. labeled_comments.json → garder: comment_id, text_clean, sentiment

Usage:
    python src/prepare_llm_data.py

"""

import json
from pathlib import Path
from typing import List, Dict, Any


# ============================================
# CONFIGURATION DES CHEMINS
# ============================================

BASE_DIR = Path(r"C:\Users\elham\Desktop\youtube-sentiment-analysis")

PATHS = {
    # Fichiers d'entrée
    "comments_for_llm": BASE_DIR / "data" / "processed" / "comments_for_llm.json",
    "labeled_comments": BASE_DIR / "data" / "labeled" / "labeled_comments.json",
    
    # Fichiers de sortie (minimisés)
    "comments_for_llm_min": BASE_DIR / "data" / "processed" / "comments_for_llm_minimal.json",
    "labeled_comments_min": BASE_DIR / "data" / "labeled" / "labeled_comments_minimal.json",
}


# ============================================
# FONCTIONS
# ============================================

def charger_json(chemin: Path) -> List[Dict[str, Any]]:
    """Charge un fichier JSON."""
    if not chemin.exists():
        raise FileNotFoundError(f"❌ Fichier non trouvé: {chemin}")
    
    with open(chemin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Chargé: {chemin.name} ({len(data)} commentaires)")
    return data


def sauvegarder_json(data: List[Dict[str, Any]], chemin: Path) -> None:
    """Sauvegarde en JSON."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    
    with open(chemin, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    taille_kb = chemin.stat().st_size / 1024
    print(f"💾 Sauvegardé: {chemin.name} ({len(data)} commentaires, {taille_kb:.1f} KB)")


def minimiser_pour_llm(commentaires: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cas 1: Commentaires NON labellisés
    Garde: comment_id, text_clean
    """
    minimises = []
    for comment in commentaires:
        minimises.append({
            "comment_id": comment.get("comment_id"),
            "text_clean": comment.get("text_clean")
        })
    return minimises


def minimiser_labeled(commentaires: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cas 2: Commentaires LABELLISÉS
    Garde: comment_id, text_clean, sentiment
    """
    minimises = []
    for comment in commentaires:
        minimises.append({
            "comment_id": comment.get("comment_id"),
            "text_clean": comment.get("text_clean"),
            "sentiment": comment.get("sentiment")
        })
    return minimises


def afficher_stats(data_original: List[Dict], data_minimal: List[Dict], nom: str) -> None:
    """Affiche les statistiques."""
    taille_original = len(json.dumps(data_original, ensure_ascii=False))
    taille_minimal = len(json.dumps(data_minimal, ensure_ascii=False))
    reduction = (1 - taille_minimal / taille_original) * 100
    
    print(f"\n📊 Stats pour {nom}:")
    print(f"   • Champs originaux: {list(data_original[0].keys()) if data_original else 'N/A'}")
    print(f"   • Champs gardés: {list(data_minimal[0].keys()) if data_minimal else 'N/A'}")
    print(f"   • Réduction taille: {reduction:.1f}%")


def afficher_exemple(data: List[Dict], nom: str, n: int = 2) -> None:
    """Affiche quelques exemples."""
    print(f"\n📋 Exemples de {nom}:")
    print("-" * 50)
    for i, item in enumerate(data[:n], 1):
        print(f"  [{i}] {json.dumps(item, ensure_ascii=False, indent=6)}")
    print("-" * 50)


# ============================================
# FONCTIONS PRINCIPALES
# ============================================

def preparer_comments_for_llm() -> None:
    """
    Cas 1: comments_for_llm_clean.json
    Garde: comment_id, text_clean
    """
    print("\n" + "=" * 60)
    print("🔄 CAS 1: Comments for LLM (non labellisés)")
    print("   Entrée: comments_for_llm_clean.json")
    print("   Sortie: comments_for_llm_minimal.json")
    print("   Champs gardés: comment_id, text_clean")
    print("=" * 60)
    
    data_original = charger_json(PATHS["comments_for_llm"])
    data_minimal = minimiser_pour_llm(data_original)
    
    afficher_stats(data_original, data_minimal, "comments_for_llm")
    afficher_exemple(data_minimal, "comments_for_llm_minimal")
    
    sauvegarder_json(data_minimal, PATHS["comments_for_llm_min"])


def preparer_labeled_comments() -> None:
    """
    Cas 2: labeled_comments.json
    Garde: comment_id, text_clean, sentiment
    """
    print("\n" + "=" * 60)
    print("🔄 CAS 2: Labeled Comments (gold data)")
    print("   Entrée: labeled_comments.json")
    print("   Sortie: labeled_comments_minimal.json")
    print("   Champs gardés: comment_id, text_clean, sentiment")
    print("=" * 60)
    
    data_original = charger_json(PATHS["labeled_comments"])
    data_minimal = minimiser_labeled(data_original)
    
    afficher_stats(data_original, data_minimal, "labeled_comments")
    
    # Distribution des sentiments
    sentiments = {}
    for item in data_minimal:
        s = item.get("sentiment", "N/A")
        sentiments[s] = sentiments.get(s, 0) + 1
    
    print(f"\n📊 Distribution des sentiments:")
    for sentiment, count in sorted(sentiments.items()):
        pct = count / len(data_minimal) * 100
        print(f"   • {sentiment}: {count} ({pct:.1f}%)")
    
    afficher_exemple(data_minimal, "labeled_comments_minimal")
    
    sauvegarder_json(data_minimal, PATHS["labeled_comments_min"])


def main():
    """Fonction principale."""
    print("\n" + "=" * 60)
    print("🚀 MINIMISATION DES DONNÉES POUR LLM")
    print("=" * 60)
    print(f"\n📂 Dossier projet: {BASE_DIR}")
    
    # Cas 1: Comments for LLM
    try:
        preparer_comments_for_llm()
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
    
    # Cas 2: Labeled Comments
    try:
        preparer_labeled_comments()
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
    
    # Résumé
    print("\n" + "=" * 60)
    print("✅ TERMINÉ!")
    print("=" * 60)
    print("\n📁 Fichiers générés:")
    
    for nom, chemin in PATHS.items():
        if "min" in nom and chemin.exists():
            taille = chemin.stat().st_size / 1024
            print(f"   • {chemin.name} ({taille:.1f} KB)")


if __name__ == "__main__":
    main()