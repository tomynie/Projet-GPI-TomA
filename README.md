# Projet-GPI-Tom AYDIN
Projet GPI noté
# 🧬 Projet M1 BBS : Apprentissage du Code & Bioinformatique

> **Objectif principal :** Développer nos compétences en programmation.

Pour le notebook (Google Colab) :

Aucune installation manuelle n'est nécessaire. La première cellule du notebook s'occupe d'installer automatiquement toutes les bibliothèques. Il suffit d'ouvrir le fichier dans Google Colab, d'uploader le fichier MRC dans l'environnement, de bien modifier le fil_id en fonction du fichier puis de lancer toutes les cellules dans l'ordre avec Ctrl+F9. Le modèle PDB se télécharge tout seul depuis le site RCSB.

Pour le script terminal (VS Code) :

Pour lancer le code depuis le terminal, il faut d'abord installer les bibliothèques manuellement une seule fois :
python -m pip install biopython mrcfile gdown scikit-image

Ensuite, se placer dans le dossier du projet :
cd "C:\Users\....\....

Puis lancer le script en précisant le fichier MRC :
python TA-cryo-code.py --mrc "Fichier MRC\14sep05c_00024sq_00006hl_00003es_c.mrc"

Il est aussi possible de modifier les paramètres de détection en ajoutant --pdb, --threshold ou --pixel_size à la commande. Les résultats sont sauvegardés automatiquement dans un dossier results/.

exemple : python TA-cryo-code.py --mrc "Fichier MRC\14sep05c_00024sq_00006hl_00003es_c.mrc" --pdb 6BDF --pixel_size 2.64 --threshold 0.55
