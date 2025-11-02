import pandas as pd
import itertools
import numpy as np

# === Charger le dataset ===
df = pd.read_csv(r"C:\Users\Sarah\Downloads\ai_job_market.csv", sep=",")

# On garde uniquement les colonnes intéressantes
cols = ["skills_required", "tools_preferred"]
df = df[cols].fillna("")

# === Transformer les valeurs en listes d’items ===
def split_items(x):
    return [i.strip() for i in x.split(",") if i.strip()]

transactions = [
    split_items(row["skills_required"]) + split_items(row["tools_preferred"])
    for _, row in df.iterrows()
]

# === Construire la matrice de contexte ===
items = sorted(set(it for trans in transactions for it in trans))
context_matrix = pd.DataFrame(
    [[1 if item in trans else 0 for item in items] for trans in transactions],
    columns=items
)

print("=== MATRICE DE CONTEXTE ===")
print(context_matrix)
context_matrix.to_csv("matrice_contexte.csv", index=False)

# === Calcul automatique du min_sup (méthode hybride) ===
item_supports = {
    item: sum(1 for t in transactions if item in t)
    for item in items
}

avg_len = np.mean(list(item_supports.values()))
std_sup = np.std(list(item_supports.values()))
cv = std_sup / avg_len  # coefficient de variation (mesure de dispersion)
facteur = min(1, max(0.1, cv))  # borne entre 0.1 et 1
min_sup_count = int((avg_len + facteur * std_sup) / 2)

print(f"\n=== CALCUL AUTOMATIQUE DU SUPPORT MINIMAL ===")
print(f"Longueur moyenne des transactions : {avg_len:.2f}")
print(f"→ seuil des transactions = {min_sup_count} transactions")
print(f"→ Support minimal automatique hybride = {min_sup_count / len(transactions):.4f}")

# === Fonction pour compter le support ===
def support_count(itemset, transactions):
    return sum(1 for t in transactions if all(i in t for i in itemset))

# === Implémentation de l’algorithme Apriori ===
def apriori(transactions, min_sup_count):
    items = sorted(set(it for trans in transactions for it in trans))
    freq_itemsets = []
    k = 1

    current_itemsets = [{item} for item in items]

    while current_itemsets:
        valid_itemsets = []
        for itemset in current_itemsets:
            sup = support_count(itemset, transactions)
            if sup >= min_sup_count:
                freq_itemsets.append((itemset, sup))
                valid_itemsets.append(itemset)

        # Génération des combinaisons k+1
        next_itemsets = [
            i.union(j)
            for i in valid_itemsets
            for j in valid_itemsets
            if len(i.union(j)) == k + 1
        ]
        current_itemsets = list(map(set, set(map(frozenset, next_itemsets))))
        k += 1

    return freq_itemsets

freq_itemsets = apriori(transactions, min_sup_count)

print("\n=== ITEMSETS FRÉQUENTS ===")
for s, sup in freq_itemsets:
    print(f"{set(s)} - support = {sup}")

# === Génération des règles d’association ===
rules = []
for itemset, sup_xy in freq_itemsets:
    if len(itemset) >= 2:
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = set(antecedent)
                consequent = itemset - antecedent
                sup_x = support_count(antecedent, transactions)
                confidence = sup_xy / sup_x if sup_x != 0 else 0
                deni = 1 - confidence
                rules.append((antecedent, consequent, sup_xy, confidence, deni))

print("\n=== RÈGLES D'ASSOCIATION ===")
for ant, cons, sup, conf, deni in rules:
    print(f"{ant} → {cons} | support={sup}, confiance={conf:.2f}, déni={deni:.2f}")

# === Sauvegarde des résultats ===
pd.DataFrame(
    [{'Itemset': list(s), 'Support': sup} for s, sup in freq_itemsets]
).to_csv("itemsets_frequents.csv", index=False)

pd.DataFrame(
    [{
        'Antécédent': list(a),
        'Conséquent': list(c),
        'Support': sup,
        'Confiance': conf,
        'Déni': deni
    } for a, c, sup, conf, deni in rules]
).to_csv("regles_association.csv", index=False)





import pandas as pd

# Charger la matrice
df = pd.read_csv('matrice_contexte.csv')

# Créer ARFF avec SEULEMENT les attributs où il y a un 1
# On ne met QUE les colonnes qui ont au moins un 1
with open('ai_job_market_true_only.arff', 'w', encoding='utf-8') as f:
    f.write("@RELATION ai_job_market_true_only\n\n")
    
    # Chaque attribut peut uniquement être "t" (ou missing "?")
    for col in df.columns:
        col_escaped = col.replace("'", "''").replace("+", "plus")
        f.write(f"@ATTRIBUTE '{col_escaped}' {{t}}\n")
    
    f.write("\n@DATA\n")
    
    # Pour chaque transaction : t si présent, ? sinon
    for _, row in df.iterrows():
        values = ['t' if val == 1 else '?' for val in row]
        f.write(",".join(values) + "\n")

print("✅ ARFF true-only créé: ai_job_market_true_only.arff")


import pandas as pd
import re

# Texte Weka avec format =t
weka_rules_text = """
 1. Reinforcement Learning=t 414 ==> TensorFlow=t 180    <conf:(0.43)> lift:(1.05) lev:(0) [9] conv:(1.03)
  2. Keras=t 406 ==> Scikit-learn=t 175    <conf:(0.43)> lift:(1.08) lev:(0.01) [13] conv:(1.05)
  3. Hugging Face=t 408 ==> LangChain=t 174    <conf:(0.43)> lift:(1.05) lev:(0) [8] conv:(1.03)
  4. Pandas=t 427 ==> FastAPI=t 181    <conf:(0.42)> lift:(1.04) lev:(0) [7] conv:(1.03)
  5. GCP=t 404 ==> TensorFlow=t 169    <conf:(0.42)> lift:(1.01) lev:(0) [2] conv:(1.01)
  6. R=t 393 ==> LangChain=t 164    <conf:(0.42)> lift:(1.03) lev:(0) [4] conv:(1.02)
  7. Azure=t 413 ==> LangChain=t 172    <conf:(0.42)> lift:(1.03) lev:(0) [4] conv:(1.02)
  8. AWS=t 404 ==> LangChain=t 168    <conf:(0.42)> lift:(1.03) lev:(0) [4] conv:(1.01)
  9. Reinforcement Learning=t 414 ==> FastAPI=t 172    <conf:(0.42)> lift:(1.02) lev:(0) [3] conv:(1.01)
 10. Pandas=t 427 ==> TensorFlow=t 177    <conf:(0.41)> lift:(1) lev:(0) [0] conv:(1)
 11. Excel=t 432 ==> Scikit-learn=t 179    <conf:(0.41)> lift:(1.04) lev:(0) [7] conv:(1.02)
 12. Python=t 402 ==> MLflow=t 165    <conf:(0.41)> lift:(1.01) lev:(0) [1] conv:(1)
 13. Hugging Face=t 408 ==> Scikit-learn=t 167    <conf:(0.41)> lift:(1.03) lev:(0) [4] conv:(1.01)
 14. Azure=t 413 ==> Scikit-learn=t 169    <conf:(0.41)> lift:(1.03) lev:(0) [4] conv:(1.01)
 15. Power BI=t 404 ==> LangChain=t 165    <conf:(0.41)> lift:(1.01) lev:(0) [1] conv:(1)
 16. CUDA=t 397 ==> FastAPI=t 162    <conf:(0.41)> lift:(1.01) lev:(0) [0] conv:(1)
 17. Cplusplus=t 390 ==> FastAPI=t 159    <conf:(0.41)> lift:(1) lev:(0) [0] conv:(1)
 18. Flask=t 398 ==> FastAPI=t 161    <conf:(0.4)> lift:(1) lev:(-0) [0] conv:(0.99)
 19. SQL=t 408 ==> LangChain=t 165    <conf:(0.4)> lift:(1) lev:(-0) [0] conv:(0.99)
 20. Keras=t 406 ==> TensorFlow=t 164    <conf:(0.4)> lift:(0.98) lev:(-0) [-3] conv:(0.98)
 21. AWS=t 404 ==> FastAPI=t 163    <conf:(0.4)> lift:(0.99) lev:(-0) [-1] conv:(0.99)
 22. Excel=t 432 ==> PyTorch=t 174    <conf:(0.4)> lift:(1.02) lev:(0) [4] conv:(1.01)
 23. R=t 393 ==> TensorFlow=t 158    <conf:(0.4)> lift:(0.97) lev:(-0) [-4] conv:(0.98)
 24. NumPy=t 416 ==> Scikit-learn=t 167    <conf:(0.4)> lift:(1.01) lev:(0) [1] conv:(1)
 25. CUDA=t 397 ==> TensorFlow=t 159    <conf:(0.4)> lift:(0.97) lev:(-0) [-4] conv:(0.98)
 26. Python=t 402 ==> Scikit-learn=t 161    <conf:(0.4)> lift:(1.01) lev:(0) [1] conv:(1)
 27. GCP=t 404 ==> Scikit-learn=t 161    <conf:(0.4)> lift:(1) lev:(0) [0] conv:(1)
 28. Power BI=t 404 ==> MLflow=t 161    <conf:(0.4)> lift:(0.98) lev:(-0) [-2] conv:(0.98)
 29. Power BI=t 404 ==> Scikit-learn=t 161    <conf:(0.4)> lift:(1) lev:(0) [0] conv:(1)
 30. Excel=t 432 ==> TensorFlow=t 172    <conf:(0.4)> lift:(0.97) lev:(-0) [-6] conv:(0.97)
 31. Power BI=t 404 ==> TensorFlow=t 160    <conf:(0.4)> lift:(0.96) lev:(-0) [-6] conv:(0.97)
 32. Python=t 402 ==> LangChain=t 159    <conf:(0.4)> lift:(0.98) lev:(-0) [-3] conv:(0.98)
 33. GCP=t 404 ==> PyTorch=t 159    <conf:(0.39)> lift:(1) lev:(0) [0] conv:(1)
 34. FastAPI=t 812 ==> TensorFlow=t 319    <conf:(0.39)> lift:(0.95) lev:(-0.01) [-15] conv:(0.97)
 35. SQL=t 408 ==> MLflow=t 160    <conf:(0.39)> lift:(0.97) lev:(-0) [-5] conv:(0.97)
 36. Pandas=t 427 ==> PyTorch=t 167    <conf:(0.39)> lift:(1) lev:(-0) [0] conv:(0.99)
 37. PyTorch=t 786 ==> TensorFlow=t 307    <conf:(0.39)> lift:(0.95) lev:(-0.01) [-17] conv:(0.96)
 38. Cplusplus=t 390 ==> PyTorch=t 152    <conf:(0.39)> lift:(0.99) lev:(-0) [-1] conv:(0.99)
 39. Flask=t 398 ==> MLflow=t 155    <conf:(0.39)> lift:(0.96) lev:(-0) [-6] conv:(0.97)
 40. NumPy=t 416 ==> FastAPI=t 162    <conf:(0.39)> lift:(0.96) lev:(-0) [-6] conv:(0.97)
 41. NumPy=t 416 ==> TensorFlow=t 162    <conf:(0.39)> lift:(0.94) lev:(-0) [-9] conv:(0.96)
 42. Excel=t 432 ==> LangChain=t 168    <conf:(0.39)> lift:(0.96) lev:(-0) [-6] conv:(0.97)
 43. KDBplus=t 499 ==> LangChain=t 194    <conf:(0.39)> lift:(0.96) lev:(-0) [-8] conv:(0.97)
 44. AWS=t 404 ==> TensorFlow=t 157    <conf:(0.39)> lift:(0.94) lev:(-0) [-9] conv:(0.96)
 45. Scikit-learn=t 796 ==> LangChain=t 309    <conf:(0.39)> lift:(0.96) lev:(-0.01) [-13] conv:(0.97)
 46. MLflow=t Scikit-learn=t 294 ==> FastAPI=t 114    <conf:(0.39)> lift:(0.96) lev:(-0) [-5] conv:(0.96)
 47. Hugging Face=t 408 ==> TensorFlow=t 158    <conf:(0.39)> lift:(0.94) lev:(-0.01) [-10] conv:(0.95)
 48. TensorFlow=t 825 ==> FastAPI=t 319    <conf:(0.39)> lift:(0.95) lev:(-0.01) [-15] conv:(0.97)
 49. Reinforcement Learning=t 414 ==> MLflow=t 160    <conf:(0.39)> lift:(0.95) lev:(-0) [-7] conv:(0.97)
 50. Pandas=t 427 ==> LangChain=t 165    <conf:(0.39)> lift:(0.95) lev:(-0) [-7] conv:(0.97)
 51. Pandas=t 427 ==> MLflow=t 165    <conf:(0.39)> lift:(0.95) lev:(-0) [-8] conv:(0.97)
 52. Pandas=t 427 ==> Scikit-learn=t 165    <conf:(0.39)> lift:(0.97) lev:(-0) [-4] conv:(0.98)
 53. MLflow=t 811 ==> FastAPI=t 313    <conf:(0.39)> lift:(0.95) lev:(-0.01) [-16] conv:(0.97)
 54. Python=t 402 ==> PyTorch=t 155    <conf:(0.39)> lift:(0.98) lev:(-0) [-2] conv:(0.98)
 55. FastAPI=t 812 ==> MLflow=t 313    <conf:(0.39)> lift:(0.95) lev:(-0.01) [-16] conv:(0.97)
 56. CUDA=t 397 ==> MLflow=t 153    <conf:(0.39)> lift:(0.95) lev:(-0) [-7] conv:(0.96)
 57. Excel=t 432 ==> FastAPI=t 166    <conf:(0.38)> lift:(0.95) lev:(-0) [-9] conv:(0.96)
 58. Keras=t 406 ==> FastAPI=t 156    <conf:(0.38)> lift:(0.95) lev:(-0) [-8] conv:(0.96)
 59. Power BI=t 404 ==> FastAPI=t 155    <conf:(0.38)> lift:(0.94) lev:(-0) [-9] conv:(0.96)
 60. GCP=t 404 ==> MLflow=t 155    <conf:(0.38)> lift:(0.95) lev:(-0) [-8] conv:(0.96)
 61. Cplusplus=t 390 ==> TensorFlow=t 149    <conf:(0.38)> lift:(0.93) lev:(-0.01) [-11] conv:(0.95)
 62. Reinforcement Learning=t 414 ==> PyTorch=t 158    <conf:(0.38)> lift:(0.97) lev:(-0) [-4] conv:(0.98)
 63. LangChain=t 810 ==> Scikit-learn=t 309    <conf:(0.38)> lift:(0.96) lev:(-0.01) [-13] conv:(0.97)
 64. PyTorch=t 786 ==> MLflow=t 299    <conf:(0.38)> lift:(0.94) lev:(-0.01) [-19] conv:(0.96)
 65. Azure=t 413 ==> TensorFlow=t 157    <conf:(0.38)> lift:(0.92) lev:(-0.01) [-13] conv:(0.94)
 66. FastAPI=t PyTorch=t 279 ==> TensorFlow=t 106    <conf:(0.38)> lift:(0.92) lev:(-0) [-9] conv:(0.94)
 67. SQL=t 408 ==> FastAPI=t 155    <conf:(0.38)> lift:(0.94) lev:(-0.01) [-10] conv:(0.95)
 68. SQL=t 408 ==> TensorFlow=t 155    <conf:(0.38)> lift:(0.92) lev:(-0.01) [-13] conv:(0.94)
 69. Excel=t 432 ==> MLflow=t 164    <conf:(0.38)> lift:(0.94) lev:(-0.01) [-11] conv:(0.95)
 70. Cplusplus=t 390 ==> MLflow=t 148    <conf:(0.38)> lift:(0.94) lev:(-0.01) [-10] conv:(0.95)
 71. Flask=t 398 ==> LangChain=t 151    <conf:(0.38)> lift:(0.94) lev:(-0.01) [-10] conv:(0.95)
 72. Keras=t 406 ==> MLflow=t 154    <conf:(0.38)> lift:(0.94) lev:(-0.01) [-10] conv:(0.95)
 73. PyTorch=t 786 ==> Scikit-learn=t 298    <conf:(0.38)> lift:(0.95) lev:(-0.01) [-14] conv:(0.97)
 74. R=t 393 ==> PyTorch=t 149    <conf:(0.38)> lift:(0.96) lev:(-0) [-5] conv:(0.97)
 75. LangChain=t 810 ==> MLflow=t 307    <conf:(0.38)> lift:(0.93) lev:(-0.01) [-21] conv:(0.96)
 76. FastAPI=t Scikit-learn=t 301 ==> MLflow=t 114    <conf:(0.38)> lift:(0.93) lev:(-0) [-8] conv:(0.95)
 77. MLflow=t 811 ==> LangChain=t 307    <conf:(0.38)> lift:(0.93) lev:(-0.01) [-21] conv:(0.96)
 78. BigQuery=t 494 ==> TensorFlow=t 187    <conf:(0.38)> lift:(0.92) lev:(-0.01) [-16] conv:(0.94)
 79. MLflow=t TensorFlow=t 304 ==> FastAPI=t 115    <conf:(0.38)> lift:(0.93) lev:(-0) [-8] conv:(0.95)
 80. Scikit-learn=t 796 ==> FastAPI=t 301    <conf:(0.38)> lift:(0.93) lev:(-0.01) [-22] conv:(0.95)
 81. Scikit-learn=t 796 ==> TensorFlow=t 301    <conf:(0.38)> lift:(0.92) lev:(-0.01) [-27] conv:(0.94)
 82. Azure=t 413 ==> PyTorch=t 156    <conf:(0.38)> lift:(0.96) lev:(-0) [-6] conv:(0.97)
 83. Hugging Face=t 408 ==> MLflow=t 154    <conf:(0.38)> lift:(0.93) lev:(-0.01) [-11] conv:(0.95)
 84. Cplusplus=t 390 ==> Scikit-learn=t 147    <conf:(0.38)> lift:(0.95) lev:(-0) [-8] conv:(0.96)
 85. Flask=t 398 ==> PyTorch=t 150    <conf:(0.38)> lift:(0.96) lev:(-0) [-6] conv:(0.97)
 86. KDBplus=t 499 ==> PyTorch=t 188    <conf:(0.38)> lift:(0.96) lev:(-0) [-8] conv:(0.97)
 87. AWS=t 404 ==> MLflow=t 152    <conf:(0.38)> lift:(0.93) lev:(-0.01) [-11] conv:(0.95)
 88. Python=t 402 ==> FastAPI=t 151    <conf:(0.38)> lift:(0.93) lev:(-0.01) [-12] conv:(0.95)
 89. Python=t 402 ==> TensorFlow=t 151    <conf:(0.38)> lift:(0.91) lev:(-0.01) [-14] conv:(0.94)
 90. CUDA=t 397 ==> PyTorch=t 149    <conf:(0.38)> lift:(0.95) lev:(-0) [-7] conv:(0.97)
 91. SQL=t 408 ==> Scikit-learn=t 153    <conf:(0.38)> lift:(0.94) lev:(-0) [-9] conv:(0.96)
 92. MLflow=t 811 ==> TensorFlow=t 304    <conf:(0.37)> lift:(0.91) lev:(-0.02) [-30] conv:(0.94)
 93. MLflow=t PyTorch=t 299 ==> TensorFlow=t 112    <conf:(0.37)> lift:(0.91) lev:(-0.01) [-11] conv:(0.93)
 94. Keras=t 406 ==> PyTorch=t 152    <conf:(0.37)> lift:(0.95) lev:(-0) [-7] conv:(0.97)
 95. Scikit-learn=t 796 ==> PyTorch=t 298    <conf:(0.37)> lift:(0.95) lev:(-0.01) [-14] conv:(0.97)
 96. LangChain=t 810 ==> FastAPI=t 303    <conf:(0.37)> lift:(0.92) lev:(-0.01) [-25] conv:(0.95)
 97. GCP=t 404 ==> LangChain=t 151    <conf:(0.37)> lift:(0.92) lev:(-0.01) [-12] conv:(0.95)
 98. FastAPI=t 812 ==> LangChain=t 303    <conf:(0.37)> lift:(0.92) lev:(-0.01) [-25] conv:(0.95)
 99. TensorFlow=t 825 ==> PyTorch=t 307    <conf:(0.37)> lift:(0.95) lev:(-0.01) [-17] conv:(0.96)
100. Flask=t 398 ==> Scikit-learn=t 148    <conf:(0.37)> lift:(0.93) lev:(-0.01) [-10] conv:(0.95)
"""

rules = []
n_trans = len(pd.read_csv('matrice_contexte.csv'))

for line in weka_rules_text.strip().split('\n'):
    if '==>' in line and '=t' in line:
        try:
            # Antécédent
            ant_part = line.split('=t')[0]
            ant = ant_part.split('. ')[-1].strip()
            
            # Conséquent
            cons_part = line.split('==>')[1].split('=t')[0]
            cons = cons_part.strip()
            
            # Support
            sup_match = re.search(r'=t\s+(\d+)\s+<', line.split('==>')[1])
            support_count = int(sup_match.group(1)) if sup_match else 0
            
            # Métriques
            conf = float(line.split('conf:(')[1].split(')')[0])
            lift = float(line.split('lift:(')[1].split(')')[0])
            
            rules.append({
                'Antécédent': ant,
                'Conséquent': cons,
                'Support': support_count / n_trans,
                'Confiance': conf,
                'Lift': lift
            })
        except Exception as e:
            continue

df_weka = pd.DataFrame(rules)
df_weka.to_csv('weka_rules.csv', index=False)
print(f"✅ {len(df_weka)} règles exportées")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# ===== CHARGEMENT =====
print("📂 Chargement des données...")
df_python = pd.read_csv('regles_association.csv')
df_weka = pd.read_csv('weka_rules.csv')
n_trans = len(pd.read_csv('matrice_contexte.csv'))

print(f"✓ Règles Python: {len(df_python)}")
print(f"✓ Règles Weka: {len(df_weka)}")
print(f"✓ Transactions: {n_trans}")

# ===== NORMALISATION =====
print("\n🔧 Normalisation des données...")

# Fonction pour clé canonique
def make_key(ant, cons):
    """Crée clé unique triée pour une règle"""
    if isinstance(ant, str):
        try:
            ant_list = ast.literal_eval(ant)
        except:
            ant_list = [ant]
    else:
        ant_list = ant if isinstance(ant, list) else [ant]
    
    if isinstance(cons, str):
        try:
            cons_list = ast.literal_eval(cons)
        except:
            cons_list = [cons]
    else:
        cons_list = cons if isinstance(cons, list) else [cons]
    
    ant_sorted = sorted([str(x).strip() for x in ant_list])
    cons_sorted = sorted([str(x).strip() for x in cons_list])
    return f"{{{','.join(ant_sorted)}}}=>{{{','.join(cons_sorted)}}}"

# Normaliser Python
df_python['key'] = df_python.apply(
    lambda x: make_key(x['Antécédent'], x['Conséquent']), axis=1
)
if df_python['Support'].max() > 1:
    df_python['Support_frac'] = df_python['Support'] / n_trans
else:
    df_python['Support_frac'] = df_python['Support']

if 'Lift' not in df_python.columns:
    df_python['Lift'] = 1.0

# Normaliser Weka
df_weka['key'] = df_weka.apply(
    lambda x: make_key(x['Antécédent'], x['Conséquent']), axis=1
)

# ===== FUSION =====
print("\n🔗 Fusion des résultats...")
df_merged = pd.merge(
    df_python[['key', 'Support_frac', 'Confiance', 'Lift']],
    df_weka[['key', 'Support', 'Confiance', 'Lift']],
    on='key',
    how='outer',
    suffixes=('_python', '_weka')
)

# ===== STATISTIQUES =====
n_python = len(df_python)
n_weka = len(df_weka)
common = df_merged.dropna(subset=['Lift_python', 'Lift_weka'])
n_common = len(common)

print(f"\n=== STATISTIQUES DE RECOUVREMENT ===")
print(f"Règles Python: {n_python}")
print(f"Règles Weka: {n_weka}")
print(f"Règles communes: {n_common}")
print(f"Uniques Python: {n_python - n_common}")
print(f"Uniques Weka: {n_weka - n_common}")
if min(n_python, n_weka) > 0:
    print(f"Taux de recouvrement: {n_common / min(n_python, n_weka) * 100:.1f}%")

# Sauvegarder
df_merged.to_csv('regles_fusionnees.csv', index=False)
print("\n✅ Résultats fusionnés: regles_fusionnees.csv")

# ===== VISUALISATION =====
print("\n📊 Génération des graphiques...")

sns.set_style("whitegrid")
fig = plt.figure(figsize=(16, 12))

has_common = len(common) > 0

# 1. Scatter Support vs Lift
ax1 = plt.subplot(2, 3, 1)
plt.scatter(df_python['Support_frac'], df_python['Lift'], 
            alpha=0.6, label='Python', s=50, color='blue')
plt.scatter(df_weka['Support'], df_weka['Lift'], 
            alpha=0.6, label='Weka', s=50, color='red', marker='^')
plt.xlabel('Support', fontsize=12)
plt.ylabel('Lift', fontsize=12)
plt.title('Support vs Lift', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 2. Histogramme Lifts
ax2 = plt.subplot(2, 3, 2)
plt.hist(df_python['Lift'].dropna(), bins=20, alpha=0.6, label='Python', color='blue')
plt.hist(df_weka['Lift'].dropna(), bins=20, alpha=0.6, label='Weka', color='red')
plt.xlabel('Lift', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.title('Distribution des Lifts', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 3. Histogramme Supports
ax3 = plt.subplot(2, 3, 3)
plt.hist(df_python['Support_frac'].dropna(), bins=20, alpha=0.6, label='Python', color='blue')
plt.hist(df_weka['Support'].dropna(), bins=20, alpha=0.6, label='Weka', color='red')
plt.xlabel('Support', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.title('Distribution des Supports', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 4. Top-N règles
ax4 = plt.subplot(2, 3, 4)
if has_common:
    n_top = min(10, len(common))
    top = common.nlargest(n_top, 'Lift_python')
    x = np.arange(len(top))
    width = 0.35
    plt.bar(x - width/2, top['Lift_python'], width, label='Python', color='blue', alpha=0.7)
    plt.bar(x + width/2, top['Lift_weka'], width, label='Weka', color='red', alpha=0.7)
    plt.xlabel('Règles', fontsize=12)
    plt.ylabel('Lift', fontsize=12)
    plt.title(f'Top-{n_top} Règles (Lift)', fontsize=14, fontweight='bold')
    plt.xticks(x, [f"R{i+1}" for i in range(len(top))], rotation=45)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Aucune règle commune', ha='center', va='center', 
             fontsize=14, color='red')
    plt.title('Top-N Règles', fontsize=14, fontweight='bold')

# 5. Corrélation Lift
ax5 = plt.subplot(2, 3, 5)
if has_common:
    plt.scatter(common['Lift_python'], common['Lift_weka'], alpha=0.6, s=50)
    max_lift = max(common['Lift_python'].max(), common['Lift_weka'].max())
    plt.plot([0, max_lift], [0, max_lift], 'r--', linewidth=2, label='y=x')
    plt.xlabel('Lift Python', fontsize=12)
    plt.ylabel('Lift Weka', fontsize=12)
    plt.title('Corrélation Lift Python vs Weka', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Aucune règle commune', ha='center', va='center', 
             fontsize=14, color='red')
    plt.title('Corrélation Lift', fontsize=14, fontweight='bold')

# 6. Recouvrement
ax6 = plt.subplot(2, 3, 6)
only_python = n_python - n_common
only_weka = n_weka - n_common
values = [only_python, n_common, only_weka]
colors = ['blue', 'purple', 'red']
categories = ['Python\nunique', 'Communes', 'Weka\nunique']

bars = plt.bar(categories, values, color=colors, alpha=0.7)
plt.ylabel('Nombre de règles', fontsize=12)
plt.title('Recouvrement des règles', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, axis='y')

for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, v + max(values)*0.02, 
             str(v), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('comparaison_apriori_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Graphiques sauvegardés: comparaison_apriori_final.png")

# ===== TABLEAU STATISTIQUES =====
stats = pd.DataFrame({
    'Métrique': ['Nombre de règles', 'Support moyen', 'Lift moyen', 'Confiance moyenne'],
    'Python': [
        n_python,
        df_python['Support_frac'].mean(),
        df_python['Lift'].dropna().mean(),
        df_python['Confiance'].mean()
    ],
    'Weka': [
        n_weka,
        df_weka['Support'].mean(),
        df_weka['Lift'].dropna().mean(),
        df_weka['Confiance'].mean()
    ]
})

stats['Python'] = stats['Python'].round(3)
stats['Weka'] = stats['Weka'].round(3)

print("\n=== STATISTIQUES COMPARATIVES ===")
print(stats.to_string(index=False))

stats.to_csv('stats_comparaison_final.csv', index=False)
print("\n✅ Stats sauvegardées: stats_comparaison_final.csv")

print("\n🎉 COMPARAISON TERMINÉE !")













