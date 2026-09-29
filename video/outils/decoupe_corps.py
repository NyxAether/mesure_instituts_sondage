# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["rembg[cpu]>=2.0", "opencv-python-headless", "numpy", "pillow"]
# ///
"""Corps en photomontage pour l'accroche : costumes découpés aux ciseaux, sans la tête (à la Karambolage).

Pour chaque corps : recadre la photo source de externe/corps/ sur la personne, détoure, retire la tête au ras du col
(la coupure de journal de externe/portraits/ viendra s'y poser), pose le corps sur le même papier que les têtes
et écrit <nom>_decoupe.png. Le point d'accroche du cou (en fraction de l'image produite) va dans cous.json.
Les PNG produits sont versionnés : le rendu Manim n'a pas besoin de ce script.

Lancement (depuis video/) : uv run --script outils/decoupe_corps.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoupe_portraits import MARGE, PAPIER, contour_ciseaux, detourage  # noqa: E402

DOSSIER = Path(__file__).resolve().parents[1] / "externe" / "corps"
HAUTEUR = 1400  # hauteur de travail du recadrage, en pixels
# Photo source, puis tout en fraction de cette photo : recadrage (gauche, haut, droite, bas), le cou (x du milieu,
# y du col, demi-largeur de la tête à retirer), et les zones à gommer (gauche, haut, droite, bas) : voisins, mobilier.
# Fillon n'a pas de photo libre en pied et seul : son corps est celui d'un militant en costume, debout à côté
# de Sarkozy sur la même photo (on ne garde pas sa tête, il n'est donc pas reconnaissable).
REGLAGES = {
    "juppe": ("juppe.jpg", (0.30, 0.07, 0.63, 1.0), (0.44, 0.265, 0.07), [(0.60, 0.55, 1.0, 1.0)]),
    "sarkozy": ("sarkozy.jpg", (0.22, 0.10, 0.62, 0.97), (0.45, 0.325, 0.075),
                [(0.0, 0.36, 0.265, 1.0), (0.0, 0.55, 0.285, 1.0), (0.54, 0.80, 1.0, 1.0)]),
    "fillon": ("sarkozy.jpg", (0.57, 0.08, 0.80, 0.97), (0.70, 0.295, 0.055), [(0.0, 0.30, 0.597, 0.62), (0.0, 0.62, 0.624, 1.0)]),
}


def decoupe(nom, rng):
    source, (g, h, d, b), (xc, yc, dl), gommes = REGLAGES[nom]
    photo = Image.open(DOSSIER / source).convert("RGB")
    W, H = photo.size
    image = photo.crop((round(g * W), round(h * H), round(d * W), round(b * H)))
    echelle = HAUTEUR / image.height
    image = image.resize((round(image.width * echelle), HAUTEUR), Image.Resampling.LANCZOS)
    # Couleurs un peu poussées, comme une photo de magazine découpée
    image = ImageEnhance.Contrast(ImageEnhance.Color(image).enhance(1.15)).enhance(1.08)

    masque = detourage(image)
    # La tête : tout ce qui dépasse le col autour du cou, le long d'un coup de ciseaux légèrement penché
    x_cou, y_cou = (xc - g) * W * echelle, (yc - h) * H * echelle
    demi = dl * W * echelle
    xs = np.arange(masque.shape[1])
    col = y_cou + 0.08 * (xs - x_cou) + rng.normal(0, 1.5, xs.size).cumsum() * 0.05
    tete = (np.abs(xs - x_cou) < demi)[None, :] & (np.arange(masque.shape[0])[:, None] < col[None, :])
    masque = masque * ~tete
    for zg, zh, zd, zb in gommes:
        x0, x1 = (max(zg, g) - g) * W * echelle, (min(zd, d) - g) * W * echelle
        y0, y1 = (max(zh, h) - h) * H * echelle, (min(zb, b) - h) * H * echelle
        masque[round(y0):round(y1), round(x0):round(x1)] = 0
    # Ne garder que le candidat (le plus grand morceau), débarrassé des voisins que le détourage aurait pris
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(masque.astype(np.uint8))
    masque = (etiquettes == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)

    pad = MARGE * 3
    masque = np.pad(masque, pad)
    rgb = np.pad(np.asarray(image, dtype=float), ((pad, pad), (pad, pad), (0, 0)))
    papier = contour_ciseaux(masque, rng)
    grain = rng.normal(0, 0.025, papier.shape)
    fond = PAPIER[None, None, :] * (1 + grain[:, :, None])
    couleur = np.where(masque[:, :, None] > 0, rgb, fond)

    ombre = cv2.GaussianBlur(np.roll(papier.astype(float), (6, 5), axis=(0, 1)), (0, 0), 5) * 0.35
    alpha = np.maximum(papier, ombre)
    couleur = np.where(papier[:, :, None] > 0, couleur, 0)
    rgba = np.dstack([np.clip(couleur, 0, 255), alpha * 255]).astype(np.uint8)

    ys, xs_ = np.where(alpha > 0.01)
    y0, x0 = ys.min(), xs_.min()
    rgba = rgba[y0:ys.max() + 1, x0:xs_.max() + 1]
    Image.fromarray(rgba, "RGBA").save(DOSSIER / f"{nom}_decoupe.png", optimize=True)
    cou = ((x_cou + pad - x0) / rgba.shape[1], (y_cou + pad - y0) / rgba.shape[0])
    print(f"{nom}_decoupe.png : {rgba.shape[1]}×{rgba.shape[0]}, cou en {cou[0]:.3f}, {cou[1]:.3f}")
    return cou


if __name__ == "__main__":
    rng = np.random.default_rng(2016)
    cous = {nom: decoupe(nom, rng) for nom in REGLAGES}
    (DOSSIER / "cous.json").write_text(json.dumps(cous, indent=2), encoding="utf-8")
