# Classification de retours citoyens avec le NLP

## Présentation

Ce projet est réalisé dans le cadre du test technique de sélection TAISS 2026 pour un stage Data Science chez Afriklang. Il consiste à classifier des commentaires citoyens sur les services publics selon trois catégories : `Satisfaction`, `Insatisfaction` et `Suggestion`.

Le dataset contient 150 commentaires équilibrés, rédigés principalement en français avec quelques insertions en éwé ou en mina. L'analyse cherche à répondre aux questions suivantes : les classes sont-elles équilibrées, quels termes les caractérisent, quel prétraitement est adapté et quel modèle généralise le mieux sur des commentaires non vus ?

## Méthode

Le notebook [notebooks/notebook.ipynb](notebooks/notebook.ipynb) suit les étapes suivantes :

1. Exploration de la structure du dataset, des catégories et de la longueur des textes.
2. Nettoyage, normalisation, gestion de quelques termes locaux, suppression des stopwords français et lemmatisation avec SpaCy.
3. Analyse du vocabulaire avec des nuages de mots et les quinze termes les plus fréquents par catégorie.
4. Séparation stratifiée en 80 % d'entraînement et 20 % de test avec `random_state=10`.
5. Vectorisation TF-IDF, puis entraînement de Naïve Bayes, d'un SVM linéaire et de Random Forest.
6. Évaluation avec l'accuracy, le F1-score macro et les matrices de confusion.

TF-IDF a été retenu car le corpus est petit, les commentaires sont courts et certains termes sont discriminants. Le vectoriseur est ajusté uniquement sur les données d'entraînement afin d'éviter une fuite d'information.

## Résultats

| Modèle | Accuracy | F1-score macro |
|---|---:|---:|
| Naïve Bayes | 56,67 % | 57,01 % |
| SVM linéaire | **66,67 %** | **66,07 %** |
| Random Forest | 50,00 % | 49,56 % |

Sur ce split, le SVM linéaire est le meilleur modèle. Cependant, les trois modèles rencontrent une difficulté persistante avec la classe `Insatisfaction`, notamment lorsqu'elle est prédite comme `Satisfaction`. Les résultats doivent être interprétés avec prudence, car le jeu de test ne contient que 30 commentaires.

## Installation et exécution

Le projet utilise `uv` et demande Python 3.13 ou une version ultérieure :

```bash
uv sync
uv run python -m spacy download fr_core_news_md
uv run jupyter lab
```

Ouvrir ensuite `notebooks/notebook.ipynb` et exécuter les cellules dans l'ordre. Le dataset attendu se trouve dans `data/dataset_nlp_test_tal.csv`.

## Limites et améliorations

Une validation croisée stratifiée permettrait de réduire la dépendance à un seul découpage. Une analyse systématique de trois à cinq erreurs, un lexique éwé/mina plus complet, des n-grammes ou des embeddings multilingues pourraient également améliorer la robustesse du pipeline.
