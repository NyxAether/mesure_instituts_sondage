"""Thème rr/ pour Manim : couleurs lues dans rr-tokens.json (tiré de la charte), polices de rr-fonts/.

Le thème se choisit avec la variable d'environnement RR_THEME (« light » par défaut, ou « dark »).
"""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

import manimpango
from manimpango import MarkupUtils
from manim import (
    DOWN,
    LEFT,
    ManimColor,
    MarkupText,
    Scene,
    VGroup,
    config,
)
from manim.mobject.text.text_mobject import START_X, START_Y, TEXT2SVG_ADJUSTMENT_FACTOR

ICI = Path(__file__).resolve().parent
TOKENS = json.loads((ICI / "rr-tokens.json").read_text(encoding="utf-8"))
THEME = os.environ.get("RR_THEME", "light")

for police in sorted((ICI.parent / "rr-fonts").glob("*.ttf")):
    manimpango.register_font(str(police))

SERIF = TOKENS["font"]["serif"]["family"]
SANS = TOKENS["font"]["sans"]["family"]
MONO = TOKENS["font"]["mono"]["family"]


class Palette:
    def __init__(self, theme):
        couleurs = TOKENS["color"][theme]
        graphique = TOKENS["chart"][theme]
        self.fond = ManimColor(couleurs["bg-primary"])
        self.carte = ManimColor(couleurs["bg-card"])
        self.texte = ManimColor(couleurs["text-primary"])
        self.texte_2 = ManimColor(couleurs["text-secondary"])
        self.discret = ManimColor(couleurs["text-muted"])
        self.accent = ManimColor(couleurs["accent"])
        self.filet = ManimColor(couleurs["rule"])
        self.grille = ManimColor(graphique["grid"])
        self.axe = ManimColor(graphique["axis"])
        self.muet = ManimColor(graphique["mark-muted"])
        self.surface = ManimColor(graphique["surface"])
        self.series = [ManimColor(c) for c in graphique["series"]]


P = Palette(THEME)
config.background_color = P.fond

# Tailles en points Manim (cadre de 8 unités de haut pour 1080 px).
TAILLE_TITRE = 52
TAILLE_TEXTE = 30
TAILLE_LIBELLE = 17
OPACITE_AIRE = 0.15
# Pango arrondit la position des lettres à la taille de rendu : aux petites tailles, la chasse
# devient irrégulière (« ti rages »). On rend donc chaque texte SURECHELLE fois plus grand, puis on le réduit.
SURECHELLE = 10
CANEVAS = 100_000  # largeur du canevas Pango, en unités Pango


def fr(x, decimales=1):
    """Nombre au format français : virgule décimale, espace fine insécable pour les milliers."""
    texte = f"{x:,.{decimales}f}".replace(",", " ").replace(".", ",")
    return texte


class TexteNet(MarkupText):
    """MarkupText rendu sur un canevas assez large pour que Pango ne revienne jamais à la ligne.

    MarkupText coupe les lignes à 500 unités Pango, ce qui casse les textes rendus en SURECHELLE.
    """

    def _text2hash(self, color):
        return super()._text2hash(color) + "-net"

    def _text2svg(self, color):
        color = ManimColor(color)
        dossier = config.get_dir("text_dir")
        dossier.mkdir(parents=True, exist_ok=True)
        fichier = dossier / (self._text2hash(color) + ".svg")
        if not fichier.exists():
            MarkupUtils.text2svg(
                f'<span foreground="{color.to_hex()}">{self.text}</span>',
                self.font,
                self.slant,
                self.weight,
                self._font_size / TEXT2SVG_ADJUSTMENT_FACTOR,
                self.line_spacing / TEXT2SVG_ADJUSTMENT_FACTOR,
                self.disable_ligatures,
                str(fichier.resolve()),
                START_X,
                START_Y,
                CANEVAS,
                CANEVAS,
                justify=self.justify,
                pango_width=None,
            )
        return str(fichier.resolve())


def ecrire(contenu, police, taille, couleur=None, markup=False):
    """Texte Pango rendu en grand puis réduit, pour une chasse régulière (markup=True : balises Pango)."""
    if not markup:
        contenu = escape(contenu)
    t = TexteNet(contenu, font=police, font_size=taille * SURECHELLE, color=couleur or P.texte)
    return t.scale(1 / SURECHELLE)


def titre(avant, mot, apres="", taille=TAILLE_TITRE):
    """Titre en Newsreader 400, un seul mot en italique prune."""
    markup = (
        f"{avant}<span font_style='italic' foreground='{P.accent.to_hex()}'>{mot}</span>{apres}"
    )
    return ecrire(markup, SERIF, taille, markup=True)


def sous_titre(contenu, taille=TAILLE_TEXTE, couleur=None):
    """Sous-titre ou annotation en Newsreader 400 (voix éditoriale de la charte)."""
    return ecrire(contenu, SERIF, taille, couleur)


def texte(contenu, taille=TAILLE_TEXTE, couleur=None):
    return ecrire(contenu, SANS, taille, couleur)


def libelle(contenu, taille=TAILLE_LIBELLE, couleur=None):
    """Libellé terminal en JetBrains Mono, minuscules."""
    return ecrire(contenu, MONO, taille, couleur or P.texte_2)


def tag_section(numero, nom):
    """Tag de section façon « // 01 · la promesse »."""
    return libelle(f"// {numero:02d} · {nom}", couleur=P.discret)


def filet_pointille(largeur, couleur=None):
    """Filet pointillé « · · · · » placé sous un titre."""
    nb = max(1, int(largeur / 0.18))
    return libelle(" ".join(["·"] * nb), couleur=couleur or P.filet)


def entete(numero, nom, avant, mot, apres=""):
    """Tag de section + titre + filet, calé en haut à gauche."""
    tag = tag_section(numero, nom)
    t = titre(avant, mot, apres)
    groupe = VGroup(tag, t).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    return groupe


class SceneRR(Scene):
    """Scène de base : fond du thème."""

    def setup(self):
        self.camera.background_color = P.fond
