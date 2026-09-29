"""Exporte en JSON les tirages et constantes de la séquence 1 (la théorie).

Les tirages numpy sont faits dans l'ordre de la scène Manim d'origine (binomiales, permutation de la population,
puis un choix de personnes par tirage montré), pour que l'histogramme reste identique à la version validée.

Lancement (depuis la racine du dépôt) : video/.venv/Scripts/python.exe video-js/export/exporter_s1.py
Écrit video-js/donnees/s1.json.
"""
import json
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent


class s1:  # noqa: N801  constantes de la séquence, reprises de la scène Manim d'origine
    GRAINE = 2016
    NB_TIRAGES = 600
    NB_RAPIDES = 40  # tirages montrés un par un (les premiers lentement, puis de plus en plus vite)
    NB_LENTS = 3  # premiers tirages montrés lentement
    LOTS = [80, 150, 300, NB_TIRAGES]  # tirages cumulés affichés ensuite, par lots
    UNITE_MAX = 0.3  # hauteur d'un tirage dans l'histogramme tant qu'il y a peu de tirages
    BORNES = np.arange(44, 56.001, 0.5)  # tranches de l'histogramme, en points de %
    Z95 = 1.96
    HAUTEUR = 3.0  # hauteur maximale d'une barre
    NB_ECHANTILLON = 110  # points mis en évidence dans la population (symbolique)
    OPACITE_AIRE = 0.15

    @staticmethod
    def marge(n, p=0.5):
        """Marge d'erreur à 95 %, en points de pourcentage."""
        return 100 * s1.Z95 * np.sqrt(p * (1 - p) / n)


TAILLE_PETIT, TAILLE_GRAND = 1000, 4000  # tailles d'échantillon des deux binomiales de la scène
P_VOTE = 0.5
COLONNES, LIGNES = 36, 22  # grille de la population dans la scène

rng = np.random.default_rng(s1.GRAINE)
t1000 = 100 * rng.binomial(TAILLE_PETIT, P_VOTE, s1.NB_TIRAGES) / TAILLE_PETIT
t4000 = 100 * rng.binomial(TAILLE_GRAND, P_VOTE, s1.NB_TIRAGES) / TAILLE_GRAND
nb = COLONNES * LIGNES
votes = rng.permutation(np.r_[np.zeros(nb // 2, int), np.ones(nb - nb // 2, int)])

# Personnes tirées pour chaque tirage montré : les NB_LENTS premiers triés (ensemble), les suivants dans l'ordre du tirage.
echantillons = []
for k in range(s1.NB_RAPIDES):
    if k < s1.NB_LENTS:
        echantillons.append(sorted(rng.choice(nb, s1.NB_ECHANTILLON, replace=False).tolist()))
    else:
        echantillons.append(rng.choice(nb, s1.NB_ECHANTILLON, replace=False).tolist())

bornes = s1.BORNES
comptes = lambda valeurs: np.histogram(valeurs, bornes)[0].tolist()  # noqa: E731
durees = np.geomspace(0.7, 0.12, s1.NB_RAPIDES - s1.NB_LENTS)  # durées des tirages rapides (copie de la scène)

sortie = {
    "source": "mêmes graine et ordre de tirage que la scène Manim d'origine",
    "graine": s1.GRAINE,
    "p": P_VOTE,
    "taille_petit": TAILLE_PETIT,
    "taille_grand": TAILLE_GRAND,
    "z95": s1.Z95,
    "nb_tirages": s1.NB_TIRAGES,
    "nb_lents": s1.NB_LENTS,
    "nb_rapides": s1.NB_RAPIDES,
    "lots": s1.LOTS,
    "unite_max": s1.UNITE_MAX,
    "opacite_aire": s1.OPACITE_AIRE,
    "hauteur": s1.HAUTEUR,
    "colonnes": COLONNES,
    "lignes": LIGNES,
    "bornes": [round(float(b), 4) for b in bornes],
    "marge_petit": round(float(s1.marge(TAILLE_PETIT, P_VOTE)), 6),
    "marge_grand": round(float(s1.marge(TAILLE_GRAND, P_VOTE)), 6),
    "durees_rapides": [round(float(d), 6) for d in durees],
    "votes": votes.tolist(),
    "t1000": [round(float(v), 4) for v in t1000],
    "t4000": [round(float(v), 4) for v in t4000],
    "echantillons": echantillons,
    # contrôles recalculés par numpy, comparés par le composant à son propre calcul des barres
    "controles": {
        "histo_40": comptes(t1000[: s1.NB_RAPIDES]),
        "histo_lots": {str(lot): comptes(t1000[:lot]) for lot in s1.LOTS},
        "histo_4000": comptes(t4000),
    },
}
assert len(votes) == nb and all(len(e) == s1.NB_ECHANTILLON for e in echantillons)
assert t1000.min() > bornes[0] and t1000.max() < bornes[-1] and t4000.min() > bornes[0] and t4000.max() < bornes[-1]
(ICI.parent / "donnees").mkdir(exist_ok=True)
(ICI.parent / "donnees" / "s1.json").write_text(json.dumps(sortie, ensure_ascii=False) + "\n", encoding="utf-8")
print("écrit", ICI.parent / "donnees" / "s1.json", "marges", sortie["marge_petit"], sortie["marge_grand"])
