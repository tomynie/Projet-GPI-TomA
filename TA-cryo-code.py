"""
Pipeline de detection de particules en cryo-EM par template matching (NCC)
==========================================================================
Utilisation :
    python pipeline.py --mrc fichier.mrc --pdb 6BDF --pixel_size 2.64 --threshold 0.55

Arguments :
    --mrc          Chemin vers le fichier MRC (optionnel : telecharge depuis Google Drive si absent)
    --pdb          Code PDB de la proteine (defaut : 6BDF)
    --pixel_size   Pixel size en A/px apres binning (defaut : 2.64)
    --threshold    Seuil de detection NCC entre 0 et 1 (defaut : 0.55)
    --bin_factor   Facteur de binning (defaut : 4)
"""

import os
import sys
import argparse
import urllib.request
import collections
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# on verifie que les bibliotheques sont bien installees
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

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
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
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def save_and_show(fig, filename, results_dir):
    """Sauvegarde la figure dans results/ et l affiche."""
    path = os.path.join(results_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   -> Sauvegarde : {path}")
    print("      (fermez la fenetre pour continuer)")
    plt.show()
    plt.close(fig)


def separator(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 1 : CHARGEMENT ET PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_mrc(mrc_path, results_dir, gdrive_id=None):
    separator("ETAPE 1 : Chargement de la micrographie")

    # priorite : fichier local -> gdrive fourni -> gdrive par defaut
    if mrc_path is not None:
        print(f"Fichier MRC local : {mrc_path}")
    else:
        file_id = gdrive_id if gdrive_id else "1Qj30jSXcHEpkzE04cisbP6ljtnQ2Ausr"
        mrc_path = "map.mrc"
        if os.path.exists(mrc_path) and os.path.getsize(mrc_path) < 1000:
            os.remove(mrc_path)
        if not os.path.exists(mrc_path):
            print(f"Telechargement depuis Google Drive (id={file_id})...")
            gdown.download(id=file_id, output=mrc_path, quiet=False)
        else:
            print(f"Fichier deja present : {mrc_path} (pas de re-telechargement)")

    with mrcfile.open(mrc_path, permissive=True) as mrc:
        raw_image = mrc.data.copy()

    image_2d = np.squeeze(raw_image)
    taille_fichier = os.path.getsize(mrc_path) / (1024 ** 2)
    print(f"Fichier    : {taille_fichier:.1f} Mo")
    print(f"Dimensions : {image_2d.shape[1]} x {image_2d.shape[0]} px  |  dtype : {image_2d.dtype}")
    print(f"Intensites : min={image_2d.min():.2f}  max={image_2d.max():.2f}  moyenne={image_2d.mean():.2f}")

    vmin, vmax = np.percentile(image_2d, [5, 95])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_2d, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Micrographie originale (brute)")
    ax.axis("off")
    save_and_show(fig, "01_micrographie_brute.png", results_dir)

    return image_2d


def preprocess(image_2d, bin_factor, results_dir):
    separator("ETAPE 1.1 : Binning")

    t0 = time.time()
    px_brut = 0.66
    px_final = px_brut * bin_factor
    print(f"Binning x{bin_factor} : {image_2d.shape[1]}x{image_2d.shape[0]} -> ", end="")
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
    pad_size = 100
    img_padded = np.pad(img_norm, pad_width=pad_size, mode="constant", constant_values=0)
    print(f"Bordure ajoutee : {pad_size}px de chaque cote (fond neutre = 0)")
    print(f"Taille finale : {img_norm.shape[1]}x{img_norm.shape[0]} -> {img_padded.shape[1]}x{img_padded.shape[0]} px")
    print(f"Duree : {time.time()-t0:.1f}s")

    img_final = img_padded.astype(np.float32)
    print("Conversion float32 OK  =>  Etape 1 terminee !")

    # affichage de l image preprocessee
    vmin, vmax = np.percentile(img_final, [5, 95])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_final, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"Image preprocessee (binning x{bin_factor}, Z-score, padding)")
    ax.axis("off")
    save_and_show(fig, "02_image_preprocessee.png", results_dir)

    return img_final


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 2 : PREPARATION DU TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def load_pdb(pdb_id):
    separator(f"ETAPE 2.1 : Import du modele PDB ({pdb_id})")

    pdb_file = f"{pdb_id}.pdb"
    if not os.path.exists(pdb_file):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Telechargement depuis RCSB : {url}")
        urllib.request.urlretrieve(url, pdb_file)
        print("Telechargement OK")
    else:
        print(f"Fichier PDB deja present : {pdb_file}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    nb_atomes = len(list(structure.get_atoms()))
    nb_residus = len(list(structure.get_residues()))
    nb_chaines = len(list(structure.get_chains()))
    print(f"Proteine    : {pdb_id}")
    print(f"Chaines     : {nb_chaines}  |  Residus : {nb_residus}  |  Atomes : {nb_atomes}")

    return structure


def create_template(structure, pixel_size, results_dir):
    separator("ETAPE 2.2 : Projection 2D du modele (template)")

    coords = np.array([atom.get_coord() for atom in structure.get_atoms()])
    coords -= coords.mean(axis=0)

    dimensions = coords.max(axis=0) - coords.min(axis=0)
    box_size = int(dimensions.max() / pixel_size * 1.2)
    if box_size % 2 != 0:
        box_size += 1

    print(f"Dimensions proteine : X={dimensions[0]:.1f}  Y={dimensions[1]:.1f}  Z={dimensions[2]:.1f} A")
    print(f"Dimension max       : {dimensions.max():.1f} A  =>  {dimensions.max()/pixel_size:.1f} px a {pixel_size} A/px")
    print(f"Box size calcule    : {box_size} px ({box_size * pixel_size:.1f} A)  [marge +20%]")

    padding_nombre = 5

    def create_projection(coord1, coord2, bins=box_size, padding=padding_nombre):
        min1, max1 = coord1.min() - padding, coord1.max() + padding
        min2, max2 = coord2.min() - padding, coord2.max() + padding
        H, _, _ = np.histogram2d(coord1, coord2, bins=bins,
                                  range=[[min1, max1], [min2, max2]])
        H = H - H.mean()
        if H.std() > 0:
            H = H / H.std()
        # inversion : atomes sombres sur fond clair (convention cryo-EM)
        H = -H
        # lissage pour simuler la resolution limitee du microscope
        H = filters.gaussian(H, sigma=1.5)
        H = (H - H.min()) / (H.max() - H.min())
        return H.astype("float32")

    template_final = create_projection(coords[:, 0], coords[:, 1])
    print(f"Template genere : {template_final.shape}, min={template_final.min():.3f}, max={template_final.max():.3f}")

    # affichage template + 3 vues
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
    save_and_show(fig, "03_template_projections.png", results_dir)

    # ajout de bruit pour visualisation + preparation pour NCC
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
    save_and_show(fig, "04_template_bruit.png", results_dir)

    # Z-score pour la NCC
    t_mean = np.mean(template_final)
    t_std = np.std(template_final)
    template_pour_ncc = ((template_final - t_mean) / (t_std + 1e-10)).astype("float32")
    print(f"Template NCC pret : mean={template_pour_ncc.mean():.3f}, std={template_pour_ncc.std():.3f}")

    return template_final, template_pour_ncc, coords


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 3 : NCC
# ─────────────────────────────────────────────────────────────────────────────

def run_ncc(img_final, template_pour_ncc, threshold, results_dir):
    separator("ETAPE 3 : NCC - Normalized Cross-Correlation")

    t0 = time.time()
    print("Calcul de la NCC sur l image complete...")
    resultat_ncc = match_template(img_final, template_pour_ncc, pad_input=True)
    print(f"NCC terminee en {time.time()-t0:.1f}s")
    print(f"Scores NCC : min={resultat_ncc.min():.3f}  max={resultat_ncc.max():.3f}  moyenne={resultat_ncc.mean():.3f}")
    print(f"(en cryo-EM un score > 0.3 est considere comme bon)")

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
    save_and_show(fig, "05_ncc_simple.png", results_dir)

    return resultat_ncc


def optimize_threshold(resultat_ncc, template_pour_ncc, img_final, threshold, results_dir):
    separator("ETAPE 3.1 : Optimisation du seuil")

    # on compare 3 seuils autour de la valeur choisie
    delta = 0.15
    seuils = [max(0.05, threshold - delta), threshold, min(0.95, threshold + delta)]
    h_t, w_t = template_pour_ncc.shape
    min_dist = h_t // 2

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
    save_and_show(fig, "06_optimisation_seuil.png", results_dir)

    pics_finaux = peak_local_max(resultat_ncc, min_distance=min_dist, threshold_rel=threshold)
    print(f"Seuil retenu : {threshold} -> {len(pics_finaux)} particules")
    return pics_finaux


def run_angular_search(img_final, template_pour_ncc, threshold, results_dir):
    separator("ETAPE 3.2 : NCC avec recherche angulaire (0-360 deg, pas 15 deg)")

    angle_step = 15
    angles = np.arange(0, 360, angle_step)
    carte_max_ncc = np.full_like(img_final, fill_value=-1.0, dtype=np.float32)
    carte_angles = np.zeros_like(img_final, dtype=np.float32)

    t_total = time.time()
    print(f"Recherche angulaire : {len(angles)} angles (pas de {angle_step} deg)...")
    for i, angle in enumerate(angles):
        t_angle = time.time()
        temp_rot = rotate(template_pour_ncc, angle, resize=False)
        ncc_angle = match_template(img_final, temp_rot, pad_input=True)
        masque = ncc_angle > carte_max_ncc
        carte_max_ncc[masque] = ncc_angle[masque]
        carte_angles[masque] = angle
        print(f"  [{i+1:2d}/{len(angles)}]  angle={angle:3d} deg  |  NCC max={ncc_angle.max():.3f}  |  {time.time()-t_angle:.1f}s", end="\r")
    print()
    print(f"Recherche terminee en {time.time()-t_total:.1f}s au total")
    print(f"Scores NCC finaux : min={carte_max_ncc.min():.3f}  max={carte_max_ncc.max():.3f}")

    h_t, w_t = template_pour_ncc.shape
    min_dist = h_t // 2
    pics_bruts = peak_local_max(carte_max_ncc, min_distance=min_dist, threshold_rel=threshold)

    # exclusion des bords (padding + demi-boite)
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
        axes[1].text(x + w_t / 2, y - h_t / 2,
                     f"{int(carte_angles[y, x])}", color="cyan", fontsize=7)
    plt.tight_layout()
    save_and_show(fig, "07_recherche_angulaire.png", results_dir)

    # bilan quantitatif
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
# ETAPE 4 : EXTRACTION DES PARTICULES
# ─────────────────────────────────────────────────────────────────────────────

def extract_particles(img_final, pics_orientes, carte_max_ncc, carte_angles,
                      template_final, template_pour_ncc, results_dir):
    separator("ETAPE 4 : Extraction des particules")

    h_box, w_box = template_pour_ncc.shape
    scores_finaux = [carte_max_ncc[y, x] for y, x in pics_orientes]
    angles_finaux = [int(carte_angles[y, x]) for y, x in pics_orientes]

    particules = []
    scores_extraits = []
    angles_extraits = []
    for (y, x), score, angle in zip(pics_orientes, scores_finaux, angles_finaux):
        y1, y2 = int(y - h_box // 2), int(y + h_box // 2)
        x1, x2 = int(x - w_box // 2), int(x + w_box // 2)
        if y1 < 0 or x1 < 0 or y2 > img_final.shape[0] or x2 > img_final.shape[1]:
            continue
        particules.append(img_final[y1:y2, x1:x2].copy())
        scores_extraits.append(score)
        angles_extraits.append(angle)

    n = len(particules)
    print(f"{n} particules extraites  ({len(pics_orientes) - n} rejetees, trop pres du bord)")

    if n == 0:
        print("Aucune particule extraite. Essayez de baisser le seuil.")
        return

    pixel_size = 2.64
    print(f"Taille de chaque boite : {h_box}x{w_box} px  =  {h_box*pixel_size:.0f}x{w_box*pixel_size:.0f} Angstroms")
    print(f"Score NCC moyen  : {np.mean(scores_extraits):.3f}")
    print(f"Score NCC max    : {np.max(scores_extraits):.3f}")
    print(f"Score NCC min    : {np.min(scores_extraits):.3f}")

    # meilleure particule
    idx_best = int(np.argmax(scores_extraits))
    meilleure_particule = particules[idx_best]
    print(f"\nMeilleure particule : #{idx_best + 1}  |  score NCC = {scores_extraits[idx_best]:.3f}  |  angle = {angles_extraits[idx_best]} deg")

    # on trie par score NCC decroissant et on garde les 12 meilleures pour la grille
    ordre = np.argsort(scores_extraits)[::-1]
    n_affiche = min(24, n)
    indices_affiches = ordre[:n_affiche]

    cols = 6
    rows = (n_affiche + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    for i, idx in enumerate(indices_affiches):
        ax = axes[i // cols][i % cols]
        ax.imshow(particules[idx], cmap="gray")
        titre = f"#{idx + 1}  |  NCC : {scores_extraits[idx]:.3f}  |  {angles_extraits[idx]}°"
        couleur = "gold" if idx == idx_best else "white"
        ax.set_title(titre, fontsize=10, color=couleur, pad=6)
        ax.axis("off")
    # cacher les cases vides
    for i in range(n_affiche, rows * cols):
        axes[i // cols][i % cols].axis("off")
    plt.suptitle(f"Top {n_affiche} particules sur {n} au total  (or = meilleure NCC)", fontsize=13)
    plt.tight_layout()
    save_and_show(fig, "08_particules_extraites.png", results_dir)

    # template vs meilleure particule
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(template_final, cmap="gray", origin="lower")
    axes[0].set_title("Template de reference")
    axes[0].axis("off")
    axes[1].imshow(meilleure_particule, cmap="gray")
    axes[1].set_title(f"Meilleure particule (score={scores_extraits[idx_best]:.3f})")
    axes[1].axis("off")
    plt.suptitle("Template vs meilleure particule detectee", fontsize=11)
    plt.tight_layout()
    save_and_show(fig, "09_meilleure_particule.png", results_dir)

    # moyenne des particules
    taille_ref = particules[0].shape
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
    save_and_show(fig, "10_moyenne_particules.png", results_dir)

    print(f"\n=> {n} particules extraites avec succes !")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # dossier de sauvegarde des figures
    results_dir = "results"
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

    # etape 1
    image_2d = load_mrc(args.mrc, results_dir, gdrive_id=args.gdrive)
    img_final = preprocess(image_2d, args.bin_factor, results_dir)

    # etape 2
    structure = load_pdb(args.pdb)
    template_final, template_pour_ncc, coords = create_template(
        structure, args.pixel_size, results_dir
    )

    # etape 3
    resultat_ncc = run_ncc(img_final, template_pour_ncc, args.threshold, results_dir)
    optimize_threshold(resultat_ncc, template_pour_ncc, img_final, args.threshold, results_dir)
    pics_orientes, carte_max_ncc, carte_angles = run_angular_search(
        img_final, template_pour_ncc, args.threshold, results_dir
    )

    # etape 4
    extract_particles(
        img_final, pics_orientes, carte_max_ncc, carte_angles,
        template_final, template_pour_ncc, results_dir
    )

    separator("PIPELINE TERMINE")
    print(f"Toutes les figures ont ete sauvegardees dans : ./{results_dir}/")
    print()
    print("Fichiers generes :")
    for f in sorted(os.listdir(results_dir)):
        taille = os.path.getsize(os.path.join(results_dir, f)) / 1024
        print(f"  {f}  ({taille:.0f} Ko)")
    print()


if __name__ == "__main__":
    main()
