# Projet-GPI-TomA
Projet GPI noté
# 🧬 Projet M1 BBS : Apprentissage du Code & Bioinformatique

> **Objectif principal :** Développer nos compétences en programmation et maîtriser la gestion de versions avec GitHub, pour les appliquer directement aux problématiques de la bioinformatique.

## 📌 Présentation du Projet

Ce dépôt a été créé dans le cadre du Master 1 BBS. Il sert de carnet de bord et d'espace de travail pour notre initiation au développement informatique. Dans le domaine de la biologie moderne, la capacité à traiter de grandes quantités de données est devenue indispensable. 

Ce projet vise à faire le pont entre nos connaissances en biologie expérimentale et l'analyse de données in silico.

### 🎯 Objectifs Visés

1. **Maîtriser les bases du code :** Écrire des scripts propres, modulaires et fonctionnels (ex: Python).
2. **Versionner avec Git et GitHub :** Assurer la traçabilité de notre travail et faciliter la collaboration.
3. **Documenter rigoureusement :** Utiliser la syntaxe *Markdown* pour rendre notre code compréhensible et reproductible.
4. **Appliquer à la Bioinformatique :** Créer des outils simples pour l'analyse de données biologiques.

---

## 🛠️ Outils & Technologies

| Catégorie      | Outil / Langage | Description |
| ----------- | ----------- | ----------- |
| **Logique**    | Python      | Écriture des scripts (`.py`) |
| **Versioning** | Git / GitHub| Suivi des modifications et sauvegarde cloud |
| **Rédaction**  | Markdown    | Formatage du présent fichier `README.md` |

## 💻 Exemple de progression

Au cours de ce projet, nous apprenons à structurer notre code pour éviter les erreurs communes (comme les boucles de récursion infinies) et à créer des modules réutilisables :
```python
# Exemple d'un module correctement structuré
def simpleprint(text):
    print(text)

if __name__ == "__main__":
    simpleprint("Test du module en local")