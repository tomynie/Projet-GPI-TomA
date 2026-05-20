"""
Pipeline de detection de particules en cryo-EM par template matching (NCC)
==========================================================================

Ce script prend une micrographie cryo-EM (fichier .mrc), construit un template
a partir d'un modele atomique (fichier PDB), puis cherche ce template dans
l'image en faisant tourner le template a differents angles. Les particules
trouvees sont extraites et sauvegardees, avec un rapport complet des resultats.

Utilisation typique :
    python TA-cryo-code.py                            -> telecharge le fichier et tourne seul
    python TA-cryo-code.py --pause                    -> s'arrete a chaque figure
    python TA-cryo-code.py --mrc mon_fichier.mrc      -> utilise un fichier local
    python TA-cryo-code.py --pdb 6BDF --threshold 0.5 -> change la proteine et le seuil

Arguments disponibles :
    --mrc          Chemin vers un fichier MRC local (optionnel)
    --gdrive       File ID Google Drive pour telecharger le MRC automatiquement
    --pdb          Code PDB de la proteine cible (defaut : 6BDF)
    --pixel_size   Taille d'un pixel en Angstroms apres binning (defaut : 2.64)
    --threshold    Seuil de detection NCC entre 0 et 1 (defaut : 0.55)
    --bin_factor   Facteur de reduction de l'image (defaut : 4)
    --pause        Afficher chaque figure et attendre avant de continuer
"""

import os
import sys
import argparse
import urllib.request
import collections
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Avant de commencer, on s'assure que toutes les bibliotheques sont la.
# Si une seule manque, on explique comment l'installer et on quitte proprement
# plutot que de crasher avec un message cryptique au milieu du pipeline.
try:
    import mrcfile
    import gdown
    from skimage import transform, filters
    from skimage.feature import match_template, peak_local_max
    from skimage.transform import rotate
    from Bio.PDB import PDBParser
except ImportError:
    print("[ERREUR] Bibliotheques manquantes. Lancez d abord :")
    print("  pip install biopython mrcfile gdown scikit-image")
    sys.exit(1)

# On fixe la graine aleatoire pour que les resultats soient reproductibles
# (le bruit ajoute au template sera toujours le meme d'une execution a l'autre)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTS EN LIGNE DE COMMANDE
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """
    Definit et lit les arguments passes au script depuis le terminal.
    Tous les arguments ont des valeurs par defaut, donc le script peut
    tourner sans rien preciser du tout.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline cryo-EM : detection de particules par template matching"
    )
    parser.add_argument(
        "--mrc", type=str, default=None,
        help="Chemin vers le fichier MRC local."
    )
    parser.add_argument(
        "--gdrive", type=str, default=None,
        help="File ID Google Drive pour telecharger le fichier MRC automatiquement."
    )
    parser.add_argument(
        "--pdb", type=str, default="6BDF",
        help="Code PDB de la proteine cible (defaut : 6BDF)"
    )
    parser.add_argument(
        "--pixel_size", type=float, default=2.64,
        help="Pixel size en Angstroms/px apres binning (defaut : 2.64)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Seuil de detection NCC entre 0 et 1 (defaut : 0.55)"
    )
    parser.add_argument(
        "--bin_factor", type=int, default=4,
        help="Facteur de binning applique a l image (defaut : 4)"
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="Afficher chaque figure et attendre avant de continuer"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def save_and_show(fig, filename, results_dir, pause=False):
    """
    Sauvegarde une figure dans le dossier de resultats.
    Si --pause est active, la fenetre s'ouvre et on attend que l'utilisateur
    la ferme avant de continuer. Sinon, on sauvegarde en silence et on passe
    a la suite sans interrompre le pipeline.
    """
    path = os.path.join(results_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   -> Sauvegarde : {path}")
    if pause:
        print("      (fermez la fenetre pour continuer)")
        plt.show()
    plt.close(fig)


def separator(title):
    """Affiche un separateur visuel dans le terminal pour bien distinguer les etapes."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 1 : CHARGEMENT DE LA MICROGRAPHIE
# ─────────────────────────────────────────────────────────────────────────────

def load_mrc(mrc_path, results_dir, gdrive_id=None, pause=False):
    """
    Charge la micrographie cryo-EM depuis un fichier .mrc.

    Si aucun fichier local n'est fourni, on essaie de le telecharger depuis
    Google Drive. On garde le fichier en cache pour eviter de le retelecharger
    a chaque execution. Si le fichier existe mais est trop petit (probablement
    un fichier HTML d'erreur), on le supprime et on retelecharge.
    """
    separator("ETAPE 1 : Chargement de la micrographie")

    # Priorite : fichier local fourni > Google Drive personnalise > Google Drive par defaut
    if mrc_path is not None:
        print(f"Fichier MRC local : {mrc_path}")
    else:
        file_id = gdrive_id if gdrive_id else "1Qj30jSXcHEpkzE04cisbP6ljtnQ2Ausr"
        mrc_path = "map.mrc"

        # Si le fichier existe mais est suspect (trop leger = probablement une erreur HTML),
        # on le supprime pour forcer un nouveau telechargement propre
        if os.path.exists(mrc_path) and os.path.getsize(mrc_path) < 1000:
            os.remove(mrc_path)

        if not os.path.exists(mrc_path):
            print(f"Telechargement depuis Google Drive (id={file_id})...")
            gdown.download(id=file_id, output=mrc_path, quiet=False)
        else:
            print(f"Fichier deja present : {mrc_path} (pas de re-telechargement)")

    # Lecture du fichier MRC avec permissive=True pour tolerer les headers non standard
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        raw_image = mrc.data.copy()

    # squeeze() elimine les dimensions superflues (ex: shape (1, H, W) -> (H, W))
    image_2d = np.squeeze(raw_image)

    taille_fichier = os.path.getsize(mrc_path) / (1024 ** 2)
    print(f"Fichier    : {taille_fichier:.1f} Mo")
    print(f"Dimensions : {image_2d.shape[1]} x {image_2d.shape[0]} px  |  dtype : {image_2d.dtype}")
    print(f"Intensites : min={image_2d.min():.2f}  max={image_2d.max():.2f}  moyenne={image_2d.mean():.2f}")

    # On utilise les percentiles 5-95 pour l'affichage afin d'eviter que quelques
    # pixels extremes ne "ecrasent" tout le contraste de l'image
    vmin, vmax = np.percentile(image_2d, [5, 95])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_2d, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Micrographie originale (brute)")
    ax.axis("off")
    save_and_show(fig, "01_micrographie_brute.png", results_dir, pause=pause)

    return image_2d


def preprocess(image_2d, bin_factor, results_dir, pause=False):
    """
    Prepare l'image pour la detection en trois sous-etapes :

    1. Binning : on reduit la resolution de l'image en moyennant des blocs de pixels.
       C'est essentiel pour que l'echelle de l'image corresponde a celle du template.
       Avec bin_factor=4 et un pixel de depart a 0.66 A/px, on arrive a 2.64 A/px,
       ce qui correspond exactement a la taille du template genere depuis le PDB.

    2. Normalisation Z-score : on centre et reduit les intensites (moyenne=0, std=1).
       C'est indispensable pour que la NCC compare des formes et non des niveaux
       de gris absolus, qui varient selon les conditions d'acquisition.

    3. Padding : on ajoute une bordure de pixels neutres (valeur 0) autour de l'image.
       Cela evite les artefacts de bord lors du calcul de la NCC, notamment les fausses
       detections dans les coins de la micrographie.
    """
    separator("ETAPE 1.1 : Binning")

    t0 = time.time()
    px_brut = 0.66  # pixel size original du microscope en Angstroms/px
    px_final = px_brut * bin_factor
    print(f"Binning x{bin_factor} : {image_2d.shape[1]}x{image_2d.shape[0]} -> ", end="")

    # downscale_local_mean est plus propre qu'un simple sous-echantillonnage :
    # il moyenne les pixels dans chaque bloc, ce qui reduit le bruit
    img_binned = transform.downscale_local_mean(image_2d, (bin_factor, bin_factor))
    print(f"{img_binned.shape[1]}x{img_binned.shape[0]} px")
    print(f"Pixel size : {px_brut} x {bin_factor} = {px_final:.2f} A/px")
    champ = img_binned.shape[1] * px_final / 10
    print(f"Champ de vue apres binning : {champ:.0f} x {img_binned.shape[0]*px_final/10:.0f} nm")
    print(f"Duree : {time.time()-t0:.1f}s")

    separator("ETAPE 1.2 : Normalisation Z-score")

    t0 = time.time()
    print(f"Avant : moyenne={img_binned.mean():.2f}  std={img_binned.std():.2f}  min={img_binned.min():.2f}  max={img_binned.max():.2f}")
    img_norm = (img_binned - img_binned.mean()) / img_binned.std()
    print(f"Apres : moyenne={img_norm.mean():.3f}  std={img_norm.std():.3f}  min={img_norm.min():.2f}  max={img_norm.max():.2f}")
    print(f"Duree : {time.time()-t0:.1f}s")

    separator("ETAPE 1.3 : Padding")

    t0 = time.time()
    pad_size = 100  # 100 pixels de marge de chaque cote
    img_padded = np.pad(img_norm, pad_width=pad_size, mode="constant", constant_values=0)
    print(f"Bordure ajoutee : {pad_size}px de chaque cote (fond neutre = 0)")
    print(f"Taille finale : {img_norm.shape[1]}x{img_norm.shape[0]} -> {img_padded.shape[1]}x{img_padded.shape[0]} px")
    print(f"Duree : {time.time()-t0:.1f}s")

    # float32 est suffisant pour la precision dont on a besoin,
    # et deux fois moins lourd en memoire que float64
    img_final = img_padded.astype(np.float32)
    print("Conversion float32 OK  =>  Etape 1 terminee !")

    vmin, vmax = np.percentile(img_final, [5, 95])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_final, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"Image preprocessee (binning x{bin_factor}, Z-score, padding)")
    ax.axis("off")
    save_and_show(fig, "02_image_preprocessee.png", results_dir, pause=pause)

    return img_final


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 2 : CONSTRUCTION DU TEMPLATE DEPUIS LE MODELE PDB
# ─────────────────────────────────────────────────────────────────────────────

def load_pdb(pdb_id):
    """
    Telecharge et parse le fichier PDB depuis la banque de donnees RCSB.
    Le fichier est garde en cache localement pour les executions suivantes.
    On affiche un resume de la proteine (nombre de chaines, residus, atomes)
    pour verifier que le bon modele a ete charge.
    """
    separator(f"ETAPE 2.1 : Import du modele PDB ({pdb_id})")

    pdb_file = f"{pdb_id}.pdb"
    if not os.path.exists(pdb_file):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Telechargement depuis RCSB : {url}")
        urllib.request.urlretrieve(url, pdb_file)
        print("Telechargement OK")
    else:
        print(f"Fichier PDB deja present : {pdb_file}")

    parser = PDBParser(QUIET=True)  # QUIET=True evite les warnings sur les residus hetero
    structure = parser.get_structure(pdb_id, pdb_file)

    nb_atomes  = len(list(structure.get_atoms()))
    nb_residus = len(list(structure.get_residues()))
    nb_chaines = len(list(structure.get_chains()))
    print(f"Proteine    : {pdb_id}")
    print(f"Chaines     : {nb_chaines}  |  Residus : {nb_residus}  |  Atomes : {nb_atomes}")

    return structure


def create_template(structure, pixel_size, results_dir, pause=False):
    """
    Construit le template 2D qui servira de reference pour la detection.

    L'idee est de simuler a quoi ressemble la proteine vue depuis le dessus
    dans une image cryo-EM, a partir de ses coordonnees atomiques 3D.

    Etapes de la construction :
    1. On recupere les coordonnees XYZ de tous les atomes et on les centre sur zero.
    2. On calcule la taille de la boite en fonction des vraies dimensions de la
       proteine en Angstroms, avec une marge de 20%.
    3. On projette les atomes sur un plan 2D via un histogramme 2D : chaque pixel
       contient le nombre d'atomes projetes dans cette region.
    4. On applique une normalisation Z-score puis on inverse le contraste : dans
       une vraie micrographie cryo-EM, les particules sont sombres sur fond clair.
       Le template doit donc avoir la meme convention pour que la NCC fonctionne.
    5. On lisse avec un filtre gaussien pour simuler la resolution limitee du
       microscope electronique (pas de structures atomiques visibles en cryo-EM).
    6. On normalise entre 0 et 1 pour l'affichage.
    7. Pour la NCC, on applique un Z-score final sur le template : c'est requis
       par l'algorithme de correlation normalisee.
    """
    separator("ETAPE 2.2 : Projection 2D du modele (template)")

    # Recuperation et centrage des coordonnees atomiques
    coords = np.array([atom.get_coord() for atom in structure.get_atoms()])
    coords -= coords.mean(axis=0)  # on centre la proteine sur l'origine

    # Calcul de la taille de la boite adaptee a la proteine
    dimensions = coords.max(axis=0) - coords.min(axis=0)
    box_size = int(dimensions.max() / pixel_size * 1.2)  # +20% de marge
    if box_size % 2 != 0:
        box_size += 1  # on s'assure que la boite est de taille paire (plus propre)

    print(f"Dimensions proteine : X={dimensions[0]:.1f}  Y={dimensions[1]:.1f}  Z={dimensions[2]:.1f} A")
    print(f"Dimension max       : {dimensions.max():.1f} A  =>  {dimensions.max()/pixel_size:.1f} px a {pixel_size} A/px")
    print(f"Box size calcule    : {box_size} px ({box_size * pixel_size:.1f} A)  [marge +20%]")

    padding_nombre = 5  # petite marge en Angstroms autour de la proteine dans l'histogramme

    def create_projection(coord1, coord2, bins=box_size, padding=padding_nombre):
        """
        Projette les atomes sur un plan 2D et genere une image simulee.
        La normalisation et l'inversion sont critiques pour que le template
        ressemble a ce qu'on voit vraiment dans la micrographie.
        """
        min1, max1 = coord1.min() - padding, coord1.max() + padding
        min2, max2 = coord2.min() - padding, coord2.max() + padding

        # Histogramme 2D : chaque bin = un pixel, sa valeur = nombre d'atomes
        H, _, _ = np.histogram2d(coord1, coord2, bins=bins,
                                  range=[[min1, max1], [min2, max2]])

        # Z-score : on centre et reduit pour que la forme prime sur les valeurs absolues
        H = H - H.mean()
        if H.std() > 0:
            H = H / H.std()

        # Inversion du contraste : en cryo-EM, les particules sont sombres sur fond clair.
        # On neglige H pour que les zones denses en atomes soient les plus sombres.
        H = -H

        # Lissage gaussien : le microscope ne resout pas les atomes individuels,
        # la densite observee est toujours "floue". sigma=1.5 px est un bon compromis.
        H = filters.gaussian(H, sigma=1.5)

        # Remise sur [0, 1] pour l'affichage et la coherence
        H = (H - H.min()) / (H.max() - H.min())
        return H.astype("float32")

    # Vue principale XY : c'est celle qui sera utilisee pour la NCC
    template_final = create_projection(coords[:, 0], coords[:, 1])
    print(f"Template genere : {template_final.shape}, min={template_final.min():.3f}, max={template_final.max():.3f}")

    # Vues complementaires XZ et YZ pour visualisation (non utilisees pour la NCC)
    proj_xz = create_projection(coords[:, 0], coords[:, 2])
    proj_yz = create_projection(coords[:, 1], coords[:, 2])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(template_final, cmap="gray", origin="lower")
    axes[0].set_title(f"Vue XY (utilisee pour NCC)\n{box_size}x{box_size} px")
    axes[0].axis("off")
    axes[1].imshow(proj_xz, cmap="gray", origin="lower")
    axes[1].set_title("Vue XZ (cote)")
    axes[1].axis("off")
    axes[2].imshow(proj_yz, cmap="gray", origin="lower")
    axes[2].set_title("Vue YZ (face)")
    axes[2].axis("off")
    plt.suptitle(f"Projections 2D de {structure.id} selon les 3 axes", fontsize=12)
    plt.tight_layout()
    save_and_show(fig, "03_template_projections.png", results_dir, pause=pause)

    # On ajoute du bruit au template juste pour montrer a quoi il ressemblerait
    # dans des conditions realistes. Ce template bruite n'est utilise que pour la figure,
    # pas pour la detection.
    niveau_de_bruit = 0.15
    bruit = np.random.normal(0.0, niveau_de_bruit, size=template_final.shape)
    template_bruit = np.clip(template_final + bruit, 0, 1).astype("float32")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(template_final, cmap="gray", origin="lower")
    axes[0].set_title("Template original")
    axes[0].axis("off")
    axes[1].imshow(template_bruit, cmap="gray", origin="lower")
    axes[1].set_title(f"Template bruite (sigma={niveau_de_bruit})")
    axes[1].axis("off")
    plt.tight_layout()
    save_and_show(fig, "04_template_bruit.png", results_dir, pause=pause)

    # Z-score final sur le template propre : necessite par match_template de skimage.
    # Le +1e-10 evite une division par zero si le template est completement plat.
    t_mean = np.mean(template_final)
    t_std  = np.std(template_final)
    template_pour_ncc = ((template_final - t_mean) / (t_std + 1e-10)).astype("float32")
    print(f"Template NCC pret : mean={template_pour_ncc.mean():.3f}, std={template_pour_ncc.std():.3f}")

    return template_final, template_pour_ncc, coords


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 3 : DETECTION PAR NCC
# ─────────────────────────────────────────────────────────────────────────────

def run_ncc(img_final, template_pour_ncc, threshold, results_dir, pause=False):
    """
    Premiere passe de detection : NCC simple avec le template dans son orientation XY.

    La NCC (Normalized Cross-Correlation) mesure la ressemblance entre le template
    et chaque region de l'image. Le score varie de -1 (anti-correlation parfaite)
    a +1 (correspondance parfaite). En cryo-EM, un score > 0.3 est generalement
    considere comme une bonne detection.

    pad_input=True garantit que la carte de correlation a la meme taille que
    l'image d'entree, ce qui facilite la correspondance des coordonnees.
    """
    separator("ETAPE 3 : NCC - Normalized Cross-Correlation")

    t0 = time.time()
    print("Calcul de la NCC sur l image complete...")
    resultat_ncc = match_template(img_final, template_pour_ncc, pad_input=True)
    print(f"NCC terminee en {time.time()-t0:.1f}s")
    print(f"Scores NCC : min={resultat_ncc.min():.3f}  max={resultat_ncc.max():.3f}  moyenne={resultat_ncc.mean():.3f}")
    print(f"(en cryo-EM un score > 0.3 est considere comme bon)")

    # min_distance = demi-taille du template : on ne peut pas avoir deux detections
    # plus proches que ca, sinon ce serait forcement la meme particule detecee deux fois
    min_dist = template_pour_ncc.shape[0] // 2
    coordonnees_pics = peak_local_max(resultat_ncc, min_distance=min_dist, threshold_rel=threshold)
    print(f"{len(coordonnees_pics)} candidats detectes  |  seuil={threshold}  |  min_distance={min_dist}px")

    h_t, w_t = template_pour_ncc.shape
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(resultat_ncc, cmap="viridis")
    axes[0].set_title("Carte de correlation (NCC)")
    axes[0].axis("off")
    axes[1].imshow(img_final, cmap="gray")
    axes[1].set_title(f"Positions detectees ({len(coordonnees_pics)} candidats)")
    axes[1].axis("off")
    for y, x in coordonnees_pics:
        rect = patches.Rectangle(
            (x - w_t // 2, y - h_t // 2), w_t, h_t,
            linewidth=1.5, edgecolor="red", facecolor="none")
        axes[1].add_patch(rect)
    plt.tight_layout()
    save_and_show(fig, "05_ncc_simple.png", results_dir, pause=pause)

    return resultat_ncc


def optimize_threshold(resultat_ncc, template_pour_ncc, img_final, threshold, results_dir, pause=False):
    """
    Compare visuellement trois seuils de detection differents.

    Le seuil choisi par l'utilisateur est teste en meme temps qu'un seuil
    plus bas (moins strict) et un seuil plus haut (plus strict). Ca permet
    de voir d'un coup d'oeil si le seuil est bien calibre ou s'il faudrait
    l'ajuster pour avoir plus ou moins de detections.
    """
    separator("ETAPE 3.1 : Optimisation du seuil")

    # On teste le seuil choisi +/- 0.15, tout en restant dans [0.05, 0.95]
    delta = 0.15
    seuils = [max(0.05, threshold - delta), threshold, min(0.95, threshold + delta)]
    h_t, w_t = template_pour_ncc.shape
    min_dist  = h_t // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, seuil in enumerate(seuils):
        pics = peak_local_max(resultat_ncc, min_distance=min_dist, threshold_rel=seuil)
        axes[i].imshow(img_final, cmap="gray")
        axes[i].set_title(f"Seuil {seuil:.2f} -> {len(pics)} particules")
        axes[i].axis("off")
        for y, x in pics:
            rect = patches.Rectangle(
                (x - w_t / 2, y - h_t / 2), w_t, h_t,
                linewidth=1, edgecolor="red", facecolor="none")
            axes[i].add_patch(rect)
    plt.tight_layout()
    save_and_show(fig, "06_optimisation_seuil.png", results_dir, pause=pause)

    pics_finaux = peak_local_max(resultat_ncc, min_distance=min_dist, threshold_rel=threshold)
    print(f"Seuil retenu : {threshold} -> {len(pics_finaux)} particules")
    return pics_finaux


def run_angular_search(img_final, template_pour_ncc, threshold, results_dir, pause=False):
    """
    Deuxieme passe de detection : NCC avec recherche angulaire complete.

    C'est l'etape la plus importante et la plus longue. Les particules dans
    une micrographie cryo-EM sont orientees aleatoirement : certaines sont
    tournees de 45°, d'autres de 180°, etc. Si on cherche uniquement avec le
    template en orientation XY, on rate toutes les particules dans d'autres
    orientations.

    On fait donc tourner le template de 0° a 345° par pas de 15° (24 angles au
    total), on calcule la NCC pour chaque angle, et on garde pour chaque pixel
    le meilleur score obtenu parmi tous les angles. On memorise egalement quel
    angle a donne ce meilleur score.

    La carte finale carte_max_ncc contient donc la meilleure correlation possible
    en tout point de l'image, quelle que soit l'orientation de la particule.

    Apres detection des pics, on exclut une marge de bord qui correspond au
    padding ajoute en etape 1 plus la demi-taille du template, pour eviter
    les fausses detections sur les bords de l'image.
    """
    separator("ETAPE 3.2 : NCC avec recherche angulaire (0-360 deg, pas 15 deg)")

    angle_step = 15
    angles = np.arange(0, 360, angle_step)

    # On initialise la carte a -1 (et non 0) pour capturer aussi les scores negatifs
    # et ne pas biaiser le resultat en faveur des scores nuls
    carte_max_ncc = np.full_like(img_final, fill_value=-1.0, dtype=np.float32)
    carte_angles  = np.zeros_like(img_final, dtype=np.float32)

    t_total = time.time()
    print(f"Recherche angulaire : {len(angles)} angles (pas de {angle_step} deg)...")

    for i, angle in enumerate(angles):
        t_angle = time.time()

        # On fait pivoter le template et on recalcule la NCC
        temp_rot  = rotate(template_pour_ncc, angle, resize=False)
        ncc_angle = match_template(img_final, temp_rot, pad_input=True)

        # Pour chaque pixel, on ne garde que le score si c'est le meilleur jusqu'ici
        masque = ncc_angle > carte_max_ncc
        carte_max_ncc[masque] = ncc_angle[masque]
        carte_angles[masque]  = angle

        print(f"  [{i+1:2d}/{len(angles)}]  angle={angle:3d} deg  |  NCC max={ncc_angle.max():.3f}  |  {time.time()-t_angle:.1f}s", end="\r")

    print()
    print(f"Recherche terminee en {time.time()-t_total:.1f}s au total")
    print(f"Scores NCC finaux : min={carte_max_ncc.min():.3f}  max={carte_max_ncc.max():.3f}")

    # Detection des pics sur la carte finale
    h_t, w_t  = template_pour_ncc.shape
    min_dist  = h_t // 2
    pics_bruts = peak_local_max(carte_max_ncc, min_distance=min_dist, threshold_rel=threshold)

    # On exclut les detections trop proches du bord :
    # - 100 px de padding ajoute en etape 1 (fond artificiel, pas de vraie particule)
    # - + demi-taille du template (la boite d'extraction depasserait du bord)
    marge_bord = 100 + h_t // 2
    H_img, W_img = img_final.shape
    pics_orientes = np.array([
        (y, x) for y, x in pics_bruts
        if marge_bord < y < H_img - marge_bord
        and marge_bord < x < W_img - marge_bord
    ])
    if len(pics_orientes) == 0:
        pics_orientes = np.empty((0, 2), dtype=int)

    print(f"-> {len(pics_orientes)} particules detectees (seuil={threshold}, bords exclus)")

    # Visualisation : carte des angles + micrographie avec les detections
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    im = axes[0].imshow(carte_angles, cmap="twilight")
    axes[0].set_title("Carte des angles optimaux")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Angle (deg)")
    axes[1].imshow(img_final, cmap="gray")
    axes[1].set_title(f"Particules detectees ({len(pics_orientes)})")
    axes[1].axis("off")
    for y, x in pics_orientes:
        rect = patches.Rectangle(
            (x - w_t / 2, y - h_t / 2), w_t, h_t,
            linewidth=1.2, edgecolor="cyan", facecolor="none")
        axes[1].add_patch(rect)
        # On affiche l'angle optimal trouve pour chaque particule
        axes[1].text(x + w_t / 2, y - h_t / 2,
                     f"{int(carte_angles[y, x])}", color="cyan", fontsize=7)
    plt.tight_layout()
    save_and_show(fig, "07_recherche_angulaire.png", results_dir, pause=pause)

    # Bilan statistique dans le terminal
    separator("Bilan quantitatif")
    if len(pics_orientes) > 0:
        scores = [carte_max_ncc[y, x] for y, x in pics_orientes]
        print(f"Particules detectees : {len(pics_orientes)}")
        print(f"Score NCC moyen      : {np.mean(scores):.3f}")
        print(f"Score NCC max        : {np.max(scores):.3f}  (meilleure detection)")
        print(f"Score NCC min        : {np.min(scores):.3f}  (detection la plus faible)")
        print(f"Ecart-type scores    : {np.std(scores):.3f}")
        print(f"\nDistribution des angles detectes :")
        comptage = collections.Counter([int(carte_angles[y, x]) for y, x in pics_orientes])
        for angle, count in sorted(comptage.items()):
            barre = "█" * count
            print(f"  {angle:3d} deg : {barre} ({count})")

    return pics_orientes, carte_max_ncc, carte_angles


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 4 : EXTRACTION ET VISUALISATION DES PARTICULES
# ─────────────────────────────────────────────────────────────────────────────

def extract_particles(img_final, pics_orientes, carte_max_ncc, carte_angles,
                      template_final, template_pour_ncc, results_dir, pause=False):
    """
    Extrait chaque particule detectee sous forme de petite image carree,
    et produit plusieurs visualisations de controle.

    Pour chaque position detectee, on decoupe une boite centree sur la particule,
    de la meme taille que le template. On rejette les particules dont la boite
    depasserait du bord de l'image (cas normalement rare grace a la marge definie
    a l'etape precedente).

    On produit ensuite :
    - Une grille avec toutes les particules triees par score NCC decroissant
      (la meilleure en or, les autres en blanc)
    - Une comparaison template vs meilleure particule
    - Une image moyenne de toutes les particules, qui estompe le bruit et fait
      ressortir la forme commune : si les detections sont bonnes, on devrait
      retrouver une forme proche du template.
    """
    separator("ETAPE 4 : Extraction des particules")

    h_box, w_box = template_pour_ncc.shape
    scores_finaux = [carte_max_ncc[y, x] for y, x in pics_orientes]
    angles_finaux = [int(carte_angles[y, x]) for y, x in pics_orientes]

    particules     = []
    scores_extraits = []
    angles_extraits = []

    for (y, x), score, angle in zip(pics_orientes, scores_finaux, angles_finaux):
        y1, y2 = int(y - h_box // 2), int(y + h_box // 2)
        x1, x2 = int(x - w_box // 2), int(x + w_box // 2)

        # On rejette les boites qui depassent les bords de l'image
        if y1 < 0 or x1 < 0 or y2 > img_final.shape[0] or x2 > img_final.shape[1]:
            continue

        particules.append(img_final[y1:y2, x1:x2].copy())
        scores_extraits.append(score)
        angles_extraits.append(angle)

    n = len(particules)
    print(f"{n} particules extraites  ({len(pics_orientes) - n} rejetees, trop pres du bord)")

    if n == 0:
        print("Aucune particule extraite. Essayez de baisser le seuil.")
        return None

    pixel_size = 2.64  # A/px apres binning x4
    print(f"Taille de chaque boite : {h_box}x{w_box} px  =  {h_box*pixel_size:.0f}x{w_box*pixel_size:.0f} Angstroms")
    print(f"Score NCC moyen  : {np.mean(scores_extraits):.3f}")
    print(f"Score NCC max    : {np.max(scores_extraits):.3f}")
    print(f"Score NCC min    : {np.min(scores_extraits):.3f}")

    # La meilleure particule est celle avec le score NCC le plus eleve
    idx_best = int(np.argmax(scores_extraits))
    meilleure_particule = particules[idx_best]
    print(f"\nMeilleure particule : #{idx_best + 1}  |  score NCC = {scores_extraits[idx_best]:.3f}  |  angle = {angles_extraits[idx_best]} deg")

    # On trie toutes les particules par score NCC decroissant pour la grille
    ordre           = np.argsort(scores_extraits)[::-1]
    n_affiche       = n
    indices_affiches = ordre[:n_affiche]

    cols = 6
    rows = (n_affiche + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # On normalise la structure axes en tableau 2D pour un acces uniforme
    # peu importe le nombre de lignes et colonnes
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]

    for i, idx in enumerate(indices_affiches):
        ax = axes[i // cols][i % cols]
        ax.imshow(particules[idx], cmap="gray")
        titre   = f"#{idx + 1}  |  NCC : {scores_extraits[idx]:.3f}  |  {angles_extraits[idx]}°"
        couleur = "gold" if idx == idx_best else "white"  # la meilleure en or
        ax.set_title(titre, fontsize=10, color=couleur, pad=6)
        ax.axis("off")

    # On cache les cases vides de la derniere ligne si le nombre de particules
    # n'est pas un multiple du nombre de colonnes
    for i in range(n_affiche, rows * cols):
        axes[i // cols][i % cols].axis("off")

    plt.suptitle(f"Toutes les particules ({n} au total)  (or = meilleure NCC)", fontsize=13)
    plt.tight_layout()
    save_and_show(fig, "08_particules_extraites.png", results_dir, pause=pause)

    # Comparaison directe : template theorique vs meilleure particule reelle
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(template_final, cmap="gray", origin="lower")
    axes[0].set_title("Template de reference")
    axes[0].axis("off")
    axes[1].imshow(meilleure_particule, cmap="gray")
    axes[1].set_title(f"Meilleure particule (score={scores_extraits[idx_best]:.3f})")
    axes[1].axis("off")
    plt.suptitle("Template vs meilleure particule detectee", fontsize=11)
    plt.tight_layout()
    save_and_show(fig, "09_meilleure_particule.png", results_dir, pause=pause)

    # Moyenne de toutes les particules : en moyennant, le bruit aleatoire s'annule
    # et la forme commune (la proteine) ressort. C'est une validation simple
    # de la qualite des detections.
    taille_ref    = particules[0].shape
    particules_ok = [p for p in particules if p.shape == taille_ref]
    moyenne_particules = np.mean(np.stack(particules_ok, axis=0), axis=0)
    print(f"Moyenne calculee sur {len(particules_ok)} particules")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(template_final, cmap="gray", origin="lower")
    axes[0].set_title("Template de reference")
    axes[0].axis("off")
    axes[1].imshow(moyenne_particules, cmap="gray")
    axes[1].set_title(f"Moyenne des {len(particules_ok)} particules extraites")
    axes[1].axis("off")
    plt.suptitle("Template vs moyenne des particules detectees", fontsize=11)
    plt.tight_layout()
    save_and_show(fig, "10_moyenne_particules.png", results_dir, pause=pause)

    print(f"\n=> {n} particules extraites avec succes !")

    # On retourne un dictionnaire de stats pour le rapport
    return {
        "n_extraites"         : n,
        "n_rejetees"          : len(pics_orientes) - n,
        "box_px"              : f"{h_box}x{w_box}",
        "box_angstrom"        : f"{h_box * pixel_size:.0f}x{w_box * pixel_size:.0f}",
        "score_moyen"         : float(np.mean(scores_extraits)),
        "score_max"           : float(np.max(scores_extraits)),
        "score_min"           : float(np.min(scores_extraits)),
        "score_std"           : float(np.std(scores_extraits)),
        "meilleure_particule" : idx_best + 1,
        "meilleur_score"      : float(scores_extraits[idx_best]),
        "meilleur_angle"      : int(angles_extraits[idx_best]),
        "distribution_angles" : dict(collections.Counter(angles_extraits)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(args, image_2d, img_final, structure, template_pour_ncc,
                    resultat_ncc, pics_orientes, carte_max_ncc, carte_angles,
                    stats_particules, duree_totale, results_dir):
    """
    Genere un fichier texte qui recapitule toutes les donnees calculees
    pendant le pipeline. Utile pour garder une trace des parametres utilises
    et des resultats obtenus, sans avoir a relire le code ou les figures.
    """

    now     = datetime.datetime.now().strftime("%d/%m/%Y a %H:%M:%S")
    px_brut = 0.66
    px_final = px_brut * args.bin_factor

    lignes = []
    def L(texte=""):
        lignes.append(texte)

    L("=" * 62)
    L("  RAPPORT D'ANALYSE  -  PIPELINE CRYO-EM (NCC)")
    L("=" * 62)
    L(f"  Genere le : {now}")
    L(f"  Duree totale du pipeline : {duree_totale:.1f} secondes")
    L()

    L("─" * 62)
    L("  PARAMETRES UTILISES")
    L("─" * 62)
    L(f"  Code PDB          : {args.pdb}")
    L(f"  Pixel size        : {args.pixel_size} A/px")
    L(f"  Facteur de binning: x{args.bin_factor}")
    L(f"  Seuil NCC         : {args.threshold}")
    L(f"  Mode pause        : {'oui' if args.pause else 'non'}")
    L()

    L("─" * 62)
    L("  ETAPE 1 — CHARGEMENT ET PREPROCESSING")
    L("─" * 62)
    L(f"  Dimensions brutes      : {image_2d.shape[1]} x {image_2d.shape[0]} px")
    L(f"  Pixel size brut        : {px_brut} A/px")
    L(f"  Apres binning x{args.bin_factor}       : {img_final.shape[1]} x {img_final.shape[0]} px")
    L(f"  Pixel size final       : {px_final:.2f} A/px")
    champ_nm_x = img_final.shape[1] * px_final / 10
    champ_nm_y = img_final.shape[0] * px_final / 10
    L(f"  Champ de vue           : {champ_nm_x:.0f} x {champ_nm_y:.0f} nm")
    L(f"  Intensite brute        : min={image_2d.min():.2f}  max={image_2d.max():.2f}  moy={image_2d.mean():.2f}")
    L(f"  Apres Z-score          : moy={img_final.mean():.3f}  std={img_final.std():.3f}")
    L(f"  Padding ajoute         : 100 px de chaque cote")
    L()

    L("─" * 62)
    L("  ETAPE 2 — PREPARATION DU TEMPLATE  (PDB : {})".format(args.pdb))
    L("─" * 62)
    nb_atomes  = len(list(structure.get_atoms()))
    nb_residus = len(list(structure.get_residues()))
    nb_chaines = len(list(structure.get_chains()))
    L(f"  Chaines               : {nb_chaines}")
    L(f"  Residus               : {nb_residus}")
    L(f"  Atomes                : {nb_atomes}")
    h_t, w_t = template_pour_ncc.shape
    L(f"  Taille du template    : {w_t} x {h_t} px  =  {w_t*args.pixel_size:.0f} x {h_t*args.pixel_size:.0f} A")
    L(f"  Template NCC          : moy={template_pour_ncc.mean():.3f}  std={template_pour_ncc.std():.3f}")
    L(f"  Lissage gaussien      : sigma = 1.5 px")
    L()

    L("─" * 62)
    L("  ETAPE 3 — NORMALIZED CROSS-CORRELATION (NCC)")
    L("─" * 62)
    L(f"  Score NCC min         : {resultat_ncc.min():.4f}")
    L(f"  Score NCC max         : {resultat_ncc.max():.4f}")
    L(f"  Score NCC moyen       : {resultat_ncc.mean():.4f}")
    L(f"  Score NCC std         : {resultat_ncc.std():.4f}")
    L()

    L("─" * 62)
    L("  ETAPE 3.2 — RECHERCHE ANGULAIRE (0-360 deg, pas 15 deg)")
    L("─" * 62)
    L(f"  Angles testes         : 24  (0 a 345 deg, pas 15 deg)")
    L(f"  Particules detectees  : {len(pics_orientes)}")
    if len(pics_orientes) > 0:
        scores_ang = [carte_max_ncc[y, x] for y, x in pics_orientes]
        L(f"  Score NCC moyen       : {np.mean(scores_ang):.4f}")
        L(f"  Score NCC max         : {np.max(scores_ang):.4f}")
        L(f"  Score NCC min         : {np.min(scores_ang):.4f}")
        L(f"  Score NCC std         : {np.std(scores_ang):.4f}")
        L()
        L("  Distribution des angles detectes :")
        comptage = collections.Counter([int(carte_angles[y, x]) for y, x in pics_orientes])
        for angle in sorted(comptage.keys()):
            count = comptage[angle]
            barre = "█" * count
            pct   = count / len(pics_orientes) * 100
            L(f"    {angle:3d} deg  {barre:<20}  {count:3d}  ({pct:.1f}%)")
    L()

    L("─" * 62)
    L("  ETAPE 4 — EXTRACTION DES PARTICULES")
    L("─" * 62)
    if stats_particules is not None:
        L(f"  Particules extraites  : {stats_particules['n_extraites']}")
        L(f"  Particules rejetees   : {stats_particules['n_rejetees']}  (trop pres du bord)")
        L(f"  Taille boite          : {stats_particules['box_px']} px  =  {stats_particules['box_angstrom']} A")
        L(f"  Score NCC moyen       : {stats_particules['score_moyen']:.4f}")
        L(f"  Score NCC max         : {stats_particules['score_max']:.4f}")
        L(f"  Score NCC min         : {stats_particules['score_min']:.4f}")
        L(f"  Score NCC std         : {stats_particules['score_std']:.4f}")
        L(f"  Meilleure particule   : #{stats_particules['meilleure_particule']}  "
          f"(score={stats_particules['meilleur_score']:.4f}, angle={stats_particules['meilleur_angle']} deg)")
        L()
        L("  Distribution des angles (particules extraites) :")
        dist  = stats_particules["distribution_angles"]
        total = sum(dist.values())
        for angle in sorted(dist.keys()):
            count = dist[angle]
            barre = "█" * count
            pct   = count / total * 100
            L(f"    {angle:3d} deg  {barre:<20}  {count:3d}  ({pct:.1f}%)")
    else:
        L("  Aucune particule extraite.")
    L()

    L("=" * 62)
    L("  FIN DU RAPPORT")
    L("=" * 62)

    rapport_path = os.path.join(results_dir, "rapport.txt")
    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    print(f"   -> Rapport sauvegarde : {rapport_path}")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTREE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestre le pipeline complet dans l'ordre logique des etapes.
    Mesure la duree totale et affiche un recapitulatif des fichiers generes
    a la fin.
    """
    args    = parse_args()
    t_debut = time.time()

    # Tous les fichiers de sortie (figures + rapport) vont dans ce dossier
    results_dir = "result-TA"
    os.makedirs(results_dir, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Pipeline cryo-EM - Detection de particules (NCC)      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  PDB         : {args.pdb}")
    print(f"  Pixel size  : {args.pixel_size} A/px")
    print(f"  Binning     : x{args.bin_factor}")
    print(f"  Seuil NCC   : {args.threshold}")
    print(f"  Resultats   : ./{results_dir}/")
    print()

    # -- Etape 1 : chargement et preparation de la micrographie
    image_2d  = load_mrc(args.mrc, results_dir, gdrive_id=args.gdrive, pause=args.pause)
    img_final = preprocess(image_2d, args.bin_factor, results_dir, pause=args.pause)

    # -- Etape 2 : construction du template depuis le modele PDB
    structure = load_pdb(args.pdb)
    template_final, template_pour_ncc, coords = create_template(
        structure, args.pixel_size, results_dir, pause=args.pause
    )

    # -- Etape 3 : detection par NCC (simple puis avec recherche angulaire)
    resultat_ncc = run_ncc(img_final, template_pour_ncc, args.threshold, results_dir, pause=args.pause)
    optimize_threshold(resultat_ncc, template_pour_ncc, img_final, args.threshold, results_dir, pause=args.pause)
    pics_orientes, carte_max_ncc, carte_angles = run_angular_search(
        img_final, template_pour_ncc, args.threshold, results_dir, pause=args.pause
    )

    # -- Etape 4 : extraction et visualisation des particules detectees
    stats_particules = extract_particles(
        img_final, pics_orientes, carte_max_ncc, carte_angles,
        template_final, template_pour_ncc, results_dir, pause=args.pause
    )

    # -- Generation du rapport texte avec toutes les donnees calculees
    duree_totale = time.time() - t_debut
    generate_report(
        args, image_2d, img_final, structure, template_pour_ncc,
        resultat_ncc, pics_orientes, carte_max_ncc, carte_angles,
        stats_particules, duree_totale, results_dir
    )

    separator("PIPELINE TERMINE")
    print(f"Duree totale : {duree_totale:.1f}s")
    print(f"Toutes les figures ont ete sauvegardees dans : ./{results_dir}/")
    print()
    print("Fichiers generes :")
    for f in sorted(os.listdir(results_dir)):
        taille = os.path.getsize(os.path.join(results_dir, f)) / 1024
        print(f"  {f}  ({taille:.0f} Ko)")
    print()


# Point d'entree : ce bloc ne s'execute que si on lance le script directement
# (pas si on l'importe depuis un autre fichier Python)
if __name__ == "__main__":
    main()
