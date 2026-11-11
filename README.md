# Analyse de Sentiment en Darija

Projet de fin d'études réalisé à l'ENSA Khouribga (filière Ingénierie des Données).

L'objectif était de construire un classifieur de sentiment (positif / négatif / neutre)
pour des commentaires en Darija marocaine collectés sur YouTube et Instagram.

La Darija, c'est compliqué pour le NLP : pas d'orthographe standard, mélange
d'arabe et de latin (arabizi), beaucoup de fautes de frappe, d'emojis, de spam.
On a donc dû construire tout le pipeline nous-mêmes, du scraping au modèle final.

---

## Contexte

On est partis d'un constat simple : les outils NLP classiques (spaCy, CamemBERT,
même les modèles arabes) ne fonctionnent pas bien sur la Darija. Les raisons :

- Pas de corpus de référence
- Orthographe non standardisée (un même mot peut s'écrire de 5 façons différentes)
- Mélange de scripts (arabe + latin + français)
- Beaucoup de bruit dans les commentaires (spam, emojis seuls, phrases génériques)

On a donc décidé de :
1. Collecter nos propres données
2. Construire un pipeline de nettoyage spécifique à la Darija
3. Labelliser manuellement un échantillon, puis automatiquement le reste
4. Comparer plusieurs modèles (classiques et deep learning)
5. Analyser les erreurs pour comprendre ce qui ne marche pas

---

## Données

### Collecte

On a utilisé l'API YouTube Data v3 pour récupérer des commentaires sur 6 chaînes
marocaines :

- Simo Sedraty (20 000 commentaires — c'est là qu'on trouve la Darija la plus riche)
- Kawalis
- Marouane53
- Raw Soueelt
- Kifache TV
- i3lamtv

Période : janvier 2025 à février 2026.

On a limité à 500 commentaires par vidéo pour éviter qu'une seule vidéo virale
biaise tout le dataset avec un vocabulaire spécifique.

En plus de YouTube, on a scrapé Instagram (commentaires sur des posts publics).

**Total brut : environ 100 000 commentaires.**

### Nettoyage

Le pipeline de nettoyage fait plusieurs choses :

- Normalisation arabe : `أإآ → ا`, `ة → ه`, suppression des diacritiques
- Réduction des élongations (`waaaaw → waaw`, on garde 2 répétitions max)
- Préservation de l'arabizi (`3`, `7`, `9`, `2` sont gardés quand ils sont collés à des lettres)
- Stopwords personnalisés (on garde les négations, sinon "ma mzyanch" devient "mzyan"...)
- Filtrage du spam (>50% de termes promotionnels)
- Suppression des phrases génériques ("ماشالله", "الحمدلله" seuls)
- Détection Darija vs MSA (arabe classique) par score

Après filtrage, on garde environ 80% des commentaires bruts.

### Labellisation

- 2 000 commentaires labellisés manuellement sur Label Studio
- 78 000 commentaires labellisés automatiquement avec Gemini
- Validation avec Label Studio + calcul du Cohen's Kappa pour l'accord inter-annotateurs

---

## Modèles testés

On a entraîné 9 modèles et tout tracké avec MLflow.

| Modèle | F1 validation | F1 test |
|---|---|---|
| SVM word+char (FeatureUnion) | 0.727 | **0.733** |
| Stacking (SVM + LR + NB) | 0.725 | **0.729** |
| Voting (SVM + LR + NB) | 0.713 | 0.720 |
| SVM TF-IDF | 0.693 | 0.691 |
| XGBoost | 0.682 | 0.681 |
| Naive Bayes | 0.667 | 0.667 |
| Logistic Regression TF-IDF | 0.658 | 0.655 |
| Random Forest | 0.646 | 0.645 |
| Logistic Regression + Embeddings | 0.585 | 0.585 |

Deux choses nous ont surpris :

1. **Le char-level TF-IDF (3-5 grammes) marche mieux que le word-level.** Logique quand
   on connaît la Darija : l'orthographe est tellement instable que les caractères
   sont plus informatifs que les mots entiers.

2. **Les modèles classiques battent le deep learning.** On a testé avec des embeddings
   SentenceTransformer + Logistic Regression, et ça donne 0.58 de F1, bien en dessous
   des SVM. Probablement parce que le dataset est encore petit et bruité.

Le meilleur modèle au final, c'est le **Stacking (SVM calibré + LR + NB)** avec
TF-IDF char_wb (3-5 grammes, max_features=30 000). F1 test : **0.7293**.

---

## Analyse d'erreurs


══════════════════════════════════════════════════════════════════════
        EXEMPLES DE CORRECTIONS HUMAINES PAR TYPE
══════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────
  TYPE : Neutre → Negatif  (1854 cas — 42.9% des erreurs)
──────────────────────────────────────────────────────────────────────

  Exemple 1 :
    Commentaire : naari fin redna
    LLM disait  : NEUTRE
    Humain a mis: NEGATIF  ← correction

  Exemple 2 :
    Commentaire : ومشي غبرات وغملات
    LLM disait  : NEUTRE
    Humain a mis: NEGATIF  ← correction

  Exemple 3 :
    Commentaire : سيمولايف ليكايمعني عليا امعني علي مو
    LLM disait  : NEUTRE
    Humain a mis: NEGATIF  ← correction

──────────────────────────────────────────────────────────────────────
  TYPE : Neutre → Positif  (1314 cas — 30.4% des erreurs)
──────────────────────────────────────────────────────────────────────

  Exemple 1 :
    Commentaire : شي ڤديو علا نفس الموضوع ولاكين من زاويه اخري وبتعمق كتر وبنفس الريتم و المتعه والحيويه اخاي سييمو
    LLM disait  : NEUTRE
    Humain a mis: POSITIF  ← correction

  Exemple 2 :
    Commentaire : حفظو القران الله يجازيك بالخير رقربات الساعه
    LLM disait  : NEUTRE
    Humain a mis: POSITIF  ← correction

  Exemple 3 :
    Commentaire : naadii dwi 3la mochkil dyal l identit
    LLM disait  : NEUTRE
    Humain a mis: POSITIF  ← correction

──────────────────────────────────────────────────────────────────────
  TYPE : Positif → Negatif  (367 cas — 8.5% des erreurs)
──────────────────────────────────────────────────────────────────────

  Exemple 1 :
    Commentaire : احسن من الدحيح
    LLM disait  : POSITIF
    Humain a mis: NEGATIF  ← correction

  Exemple 2 :
    Commentaire : a7mad concepte
    LLM disait  : POSITIF
    Humain a mis: NEGATIF  ← correction

  Exemple 3 :
    Commentaire : ki derti sberti lih maymkench hada ra khasso ja2iza dial semm w l7i9d
    LLM disait  : POSITIF
    Humain a mis: NEGATIF  ← correction

──────────────────────────────────────────────────────────────────────
  TYPE : Negatif → Neutre  (325 cas — 7.5% des erreurs)
──────────────────────────────────────────────────────────────────────

  Exemple 1 :
    Commentaire : هرباان اخي تريgل مريgل
    LLM disait  : NEGATIF
    Humain a mis: NEUTRE  ← correction

  Exemple 2 :
    Commentaire : لا نت مشي طبعي
    LLM disait  : NEGATIF
    Humain a mis: NEUTRE  ← correction

  Exemple 3 :
    Commentaire : rak tcopiyi 3la mester x samir zian yaw fa9o
    LLM disait  : NEGATIF
    Humain a mis: NEUTRE  ← correction

──────────────────────────────────────────────────────────────────────
  TYPE : Positif → Neutre  (244 cas — 5.6% des erreurs)
──────────────────────────────────────────────────────────────────────

  Exemple 1 :
    Commentaire : madghetch 3la rassek bach der lfloss dghet 3la subscribe bach ana ndirhom hh welcome back legend
    LLM disait  : POSITIF
    Humain a mis: NEUTRE  ← correction

  Exemple 2 :
    Commentaire : و فالاخير يوتيب يطلع لي اشهار علي التريدنگ بلا سدراني عوده وازنه
    LLM disait  : POSITIF
    Humain a mis: NEUTRE  ← correction

  Exemple 3 :
    Commentaire : فراسك كتشبه بزااف لاحمد حلمي فكولشي
    LLM disait  : POSITIF
    Humain a mis: NEUTRE  ← correction

──────────────────────────────────────────────────────────────────────
  TYPE : Negatif → Positif  (218 cas — 5.0% des erreurs)
──────────────────────────────────────────────────────────────────────

  Exemple 1 :
    Commentaire : هه ميمكنش نتا كون كنتي غير كتشرح لنا دروس
    LLM disait  : NEGATIF
    Humain a mis: POSITIF  ← correction

  Exemple 2 :
    Commentaire : wlh hta m9wde nta zwina hiya tatb9a dir l3aynik tathrkhom
    LLM disait  : NEGATIF
    Humain a mis: POSITIF  ← correction

  Exemple 3 :
    Commentaire : ana 3ziz 3liya simo walakine ila hiyed haja wa7da ghay wli wa3er i na9ass chwiyi men drafa wi zid fles videos
    LLM disait  : NEGATIF
    Humain a mis: POSITIF  ← correction

══════════════════════════════════════════════════════════════════════


---

Organisation du projet

darija-sentiment-analysis/
├── src/                    # code source
│   ├── scraper.py
│   ├── preprocessing.py
│   ├── prepare_llm_data.py
│   ├── normlize_lmmlab.py
│   ├── compute_agreement.py
│   ├── combainison_llm.py
│   ├── merge_full_data.py
│   ├── apply_labelstudio_changes.py
│   └── count_bad_words.py
├── notebooks/              # EDA + modélisation + analyse
│   ├── 01_EDA_Dataset_Final.ipynb
│   ├── 02_EDA_Labeled_Minimal.ipynb
│   ├── 03_final_analysis.ipynb
│   ├── 04_ml_classique.ipynb
│   └── 05_llm_vs_corrected_comparison.ipynb
├── sample/
│   └── échantillon_data/             # 100 
└──