# Projet-GPI-Tom AYDIN
# 🧬 Projet GPI — Tom Aydin
**M1 BBS · Bioinformatique · 2024–2025**

Pipeline de détection de particules en cryo-EM par template matching (NCC).

---

## 📋 Description

Ce projet implémente un pipeline complet de particle picking en cryo-microscopie électronique. À partir d'une micrographie au format `.mrc` et d'un modèle atomique PDB, le pipeline :

- Prétraite la micrographie (binning, normalisation, padding)
- Génère un template 2D depuis les coordonnées atomiques
- Détecte les particules par corrélation croisée normalisée (NCC)
- Effectue une recherche angulaire sur 360° (24 angles, pas de 15°)
- Extrait et visualise toutes les particules détectées

Le projet est disponible sous deux formes : un **notebook Jupyter** (Google Colab) et un **script Python** exécutable depuis le terminal.

---

## 📁 Fichiers

| Fichier | Description |
|---|---|
| `GPI_Aydin_Tom.ipynb` | Notebook interactif (Google Colab) |
| `TA-cryo-code.py` | Script terminal autonome |

---

## 🚀 Utilisation — Notebook (Google Colab)

Aucune installation manuelle n'est nécessaire.

1. Ouvrir `GPI_Aydin_Tom.ipynb` dans Google Colab
2. Uploader le fichier `.mrc` dans l'environnement Colab
3. Modifier le `file_id` dans la cellule de chargement selon votre fichier
4. Modifier le code PDB si besoin (défaut : `6BDF`)
5. Lancer toutes les cellules dans l'ordre

> Le modèle PDB se télécharge automatiquement depuis [RCSB](https://www.rcsb.org).

---

## 💻 Utilisation — Script Terminal (VS Code)

### 1. Installation des bibliothèques (une seule fois)

```bash
python -m pip install biopython mrcfile gdown scikit-image
Si plusieurs versions de Python sont installées, préciser : python3 -m pip install ...

2. Se placer dans le dossier du projet
cd "C:\Users\...\GPI"
3. Lancer le script
Avec un fichier MRC local :

python TA-cryo-code.py --mrc "Fichier MRC\fichier.mrc"
Avec un identifiant Google Drive :

python TA-cryo-code.py --gdrive 1Qj30jSXcHEpkzE04cisbP6ljtnQ2Ausr
Exemple complet avec tous les paramètres :

python TA-cryo-code.py --gdrive 1Qj30jSXcHEpkzE04cisbP6ljtnQ2Ausr --pdb 6BDF --pixel_size 2.64 --threshold 0.55
⚙️ Arguments disponibles
Argument	Description	Défaut
--mrc	Chemin vers un fichier MRC local	—
--gdrive	File ID Google Drive du fichier MRC	—
--pdb	Code PDB de la protéine cible	6BDF
--pixel_size	Taille d'un pixel en Å/px après binning	2.64
--threshold	Seuil de détection NCC (entre 0 et 1)	0.55
--bin_factor	Facteur de réduction de l'image	4
--pause	Afficher chaque figure avant de continuer	désactivé


📂 Résultats
Tous les fichiers générés sont sauvegardés dans le dossier result-TA/ :

Fichier	Contenu
01_micrographie_brute.png	Image MRC originale
02_image_preprocessee.png	Après binning, Z-score et padding
03_template_projections.png	Vues XY / XZ / YZ du modèle PDB
04_template_bruit.png	Template avec et sans bruit
05_ncc_simple.png	Carte NCC et détections initiales
06_optimisation_seuil.png	Comparaison de 3 seuils de détection
07_recherche_angulaire.png	Résultats avec recherche sur 360°
08_particules_extraites.png	Grille de toutes les particules
09_meilleure_particule.png	Template vs meilleure détection
10_moyenne_particules.png	Template vs moyenne des particules
rapport.txt	Rapport complet avec toutes les stats




