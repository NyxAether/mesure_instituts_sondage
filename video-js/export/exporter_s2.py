"""Exporte en JSON les chiffres de la séquence 2 (entonnoir), avec les mêmes tirages que la scène Manim.

Lancement (depuis la racine du dépôt) : video/.venv/Scripts/python.exe video-js/export/exporter_s2.py
Écrit video-js/donnees/s2.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent.parent / "video"))
from donnees import TOUS  # noqa: E402

# Mêmes constantes que la scène Manim d'origine
GRAINE = 2002
Z95 = 1.96
EXEMPLE = {"pays": "France", "annee": 2002, "n": 1000.0, "vote": 0.1686, "poll": 0.13}

nuage = TOUS["nuage"]
pts = nuage["points"]
n = np.array([p["n"] for p in pts])
vote = np.array([p["vote"] for p in pts])
residu = 100 * np.array([p["residu"] for p in pts])
marge_propre = 100 * Z95 * np.sqrt(vote * (1 - vote) / n)
# Même ordre des tirages que la scène : un seul appel à binomial sur le générateur neuf.
rng = np.random.default_rng(GRAINE)
simule = 100 * (rng.binomial(n.astype(int), vote) / n - vote)

assert np.isclose((np.abs(residu) > marge_propre).mean(), nuage["part_hors_marge"]), "part hors marge différente de l'export"
assert any(all(p[k] == v for k, v in EXEMPLE.items()) for p in pts), "point d'exemple absent des données"

sortie = {
    "source": "mesure_erreurs (base Jennings et Wlezien), via video/donnees.py:TOUS['nuage'] ; tirages : numpy.random.default_rng(graine).binomial",
    "graine": GRAINE,
    "z95": Z95,
    "annee_min": TOUS["annee_min"],
    "nb_lignes": nuage["nb_lignes"],
    "nb_sondages": nuage["nb_sondages"],
    "nb_pays": nuage["nb_pays"],
    "part_hors_marge": nuage["part_hors_marge"],
    "exemple": EXEMPLE,
    # Une entrée par intention de vote : taille, écart réel, écart d'un tirage aléatoire, marge propre (points).
    "points": [
        {"n": float(n[i]), "reel": round(float(residu[i]), 6), "simule": round(float(simule[i]), 6), "marge": round(float(marge_propre[i]), 6)}
        for i in range(len(pts))
    ],
    "marge": [{"n": m["n"], "m": m["m"]} for m in nuage["marge"]],
}
(ICI.parent / "donnees").mkdir(exist_ok=True)
(ICI.parent / "donnees" / "s2.json").write_text(json.dumps(sortie, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("écrit", ICI.parent / "donnees" / "s2.json")
