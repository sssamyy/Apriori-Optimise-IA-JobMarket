# Algorithme Apriori Optimisé – Application au Marché de l’Emploi IA

Ce projet implémente une version **optimisée de l’algorithme Apriori** avec **support minimal automatisé**, puis la compare à la version **classique sous Weka**, en appliquant les deux approches à un jeu de données sur les offres d'emploi en Intelligence Artificielle.

## 🎯 Objectif

- Automatiser le calcul du support minimal grâce à une approche **hybride basée sur la moyenne et l’écart-type**.
- Extraire et comparer les règles d’association générées par :
  - Notre **implémentation optimisée** (Python)
  - L’algorithme **Apriori classique** disponible dans **Weka**

## 📊 Données utilisées

- Données sur les **compétences, entreprises, intitulés de poste**, etc.
- Extraction des attributs pertinents pour constituer une matrice transactionnelle.

## 🧠 Fonctionnalités

- Prétraitement automatique des données (Pandas)
- Transformation en **format transactionnel** (One-Hot Encoding)
- Implémentation personnalisée de l’algorithme **Apriori**
- Export des résultats en **CSV**, **Excel**, ou **JSON**
- Conversion automatique en **ARFF** pour Weka
- Extraction et parsing des règles d’association générées par Weka
- **Visualisation comparative** des deux jeux de règles :
  - Recouvrement
  - Support / Confiance / Lift
  - Graphiques Matplotlib / Seaborn

## 📐 Automatisation du Support Minimal

Support calculé automatiquement par :
min_sup = int((mean + 0.5 * std_dev) / 2)

Méthode **adaptative** qui ajuste le support selon la densité des données.

##📈 Résultats attendus

Règles d’association pertinentes pour le marché de l’IA

Gains de temps et précision grâce au support automatisé

Visualisation claire des différences entre les deux approches

##🏁 Conclusion

Ce projet montre l’apport d’une approche hybride dans le calcul de support pour améliorer l'extraction de connaissance via l’algorithme Apriori, tout en facilitant le processus via une automatisation et des outils d’analyse.
---









