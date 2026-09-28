# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["rembg[cpu]>=2.0", "opencv-python-headless", "numpy", "pillow"]
# ///
"""Portraits façon « coupure de journal » (à la Karambolage) pour l'accroche.

Détoure chaque photo de externe/portraits/<nom>.jpg, la passe en trame de journal,
la pose sur un morceau de papier découpé aux ciseaux et écrit <nom>_decoupe.png.
Les PNG produits sont versionnés : le rendu Manim n'a pas besoin de ce script.

Lancement (depuis video/) : uv run --script outils/decoupe_portraits.py
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove

DOSSIER = Path(__file__).resolve().parents[1] / "externe" / "portraits"
# Hauteur (fraction de la hauteur du détourage, depuis le haut de la tête) où passent les ciseaux,
# et pente de la coupe (en fraction de la largeur).
COUPE = {
    "fillon": (0.80, 0.04), "juppe": (0.82, -0.05), "sarkozy": (0.80, 0.05),
    "macron": (0.78, -0.04), "lepen": (0.80, 0.05),
}
HAUTEUR = 900  # hauteur de travail, en pixels
ENCRE = np.array([34, 30, 28], dtype=float)
PAPIER = np.array([240, 234, 220], dtype=float)
MARGE = 14  # largeur du bord de papier autour du détourage, en pixels
PAS_TRAME = 18  # pas de la trame, en pixels : environ 3,5 pixels à l’écran pour une tête de 1,1 unité en 1080p
PART_TRAME = 0.8  # part de la trame face au gris continu : plus elle est forte, plus les points se voient


def detourage(image):
    alpha = np.array(remove(image))[:, :, 3]
    masque = (alpha > 128).astype(np.uint8)
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(masque)
    plus_grande = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    masque = (etiquettes == plus_grande).astype(np.uint8)
    return cv2.morphologyEx(masque, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def coupe_ciseaux(masque, hauteur, pente, rng):
    """Retire le bas du buste le long d'une ligne légèrement penchée."""
    lignes = np.where(masque.any(axis=1))[0]
    haut, bas = lignes[0], lignes[-1]
    y0 = haut + hauteur * (bas - haut)
    largeur = masque.shape[1]
    xs = np.arange(largeur)
    limite = y0 + pente * (xs - largeur / 2) + rng.normal(0, 1.5, largeur).cumsum() * 0.05
    return masque * (np.arange(masque.shape[0])[:, None] < limite[None, :])


def trame(gris):
    """Trame de points à 45° (similigravure), mêlée au gris pour garder le visage lisible."""
    h, w = gris.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u, v = (xx + yy) / np.sqrt(2), (xx - yy) / np.sqrt(2)
    cellule = np.hypot((u % PAS_TRAME) - PAS_TRAME / 2, (v % PAS_TRAME) - PAS_TRAME / 2) / (PAS_TRAME / np.sqrt(2))
    encre = 1 - gris  # 0 = blanc, 1 = noir
    points = np.clip((np.sqrt(encre) - cellule) * 4 + 0.5, 0, 1)
    return PART_TRAME * points + (1 - PART_TRAME) * encre


def contour_ciseaux(masque, rng):
    """Bord de papier : le détourage élargi, redessiné en segments droits comme des coups de ciseaux."""
    elargi = cv2.dilate(masque, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGE + 1,) * 2))
    contours, _ = cv2.findContours(elargi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(contour, 5, True)[:, 0, :].astype(float)
    poly += rng.normal(0, 2.0, poly.shape)
    papier = np.zeros_like(masque)
    cv2.fillPoly(papier, [poly.round().astype(np.int32)], 1)
    return papier


def decoupe(nom, rng):
    image = Image.open(DOSSIER / f"{nom}.jpg").convert("RGB")
    image = image.resize((round(image.width * HAUTEUR / image.height), HAUTEUR), Image.Resampling.LANCZOS)
    masque = coupe_ciseaux(detourage(image), *COUPE[nom], rng)

    gris = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(float) / 255
    # Contraste de papier journal : on étire les niveaux sur le seul visage.
    bas, haut = np.percentile(gris[masque > 0], [2, 98])
    gris = np.clip((gris - bas) / (haut - bas), 0, 1) ** 1.1
    encre = trame(gris)

    # On agrandit le canevas pour loger le bord de papier et l'ombre.
    pad = MARGE * 3
    masque = np.pad(masque, pad)
    encre = np.pad(encre, pad)
    papier = contour_ciseaux(masque, rng)

    grain = rng.normal(0, 0.025, papier.shape)
    fond = PAPIER[None, None, :] * (1 + grain[:, :, None])
    couleur = fond * (1 - encre[:, :, None] * masque[:, :, None]) + ENCRE * (encre * masque)[:, :, None]

    ombre = cv2.GaussianBlur(np.roll(papier.astype(float), (6, 5), axis=(0, 1)), (0, 0), 5) * 0.35
    alpha = np.maximum(papier, ombre)
    couleur = np.where(papier[:, :, None] > 0, couleur, 0)
    rgba = np.dstack([np.clip(couleur, 0, 255), alpha * 255]).astype(np.uint8)

    ys, xs = np.where(alpha > 0.01)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    Image.fromarray(rgba, "RGBA").save(DOSSIER / f"{nom}_decoupe.png", optimize=True)
    print(f"{nom}_decoupe.png : {rgba.shape[1]}×{rgba.shape[0]}")


if __name__ == "__main__":
    rng = np.random.default_rng(2016)
    for nom in COUPE:
        decoupe(nom, rng)
