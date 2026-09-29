"""Exporte en JSON les chiffres de la séquence 4 (taille équivalente), lus par video/donnees.py comme dans la version Manim.

Lancement (depuis la racine du dépôt) : video/.venv/Scripts/python.exe video-js/export/exporter_s4.py
Écrit video-js/donnees/s4.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent.parent / "video"))
from donnees import TOUS, glissante_equivalents  # noqa: E402

# Mêmes constantes que video/scenes/s4_equivalente.py
Z95 = 1.96
CIBLE = 0.95  # part des écarts que l'entonnoir élargi doit contenir
JOURS_LISSAGE = 7
ETAPES = [2, 4]  # facteurs montrés avant le facteur final

eq = TOUS["equivalents"]
nuage = TOUS["nuage"]
pts = nuage["points"]
n = np.array([p["n"] for p in pts])
vote = np.array([p["vote"] for p in pts])
residu = 100 * np.array([p["residu"] for p in pts])
sigma = 100 * np.sqrt(vote * (1 - vote) / n)
facteur_95 = float(np.quantile((np.abs(residu) / (Z95 * sigma)) ** 2, CIBLE))
assert np.isclose(1 - (np.abs(residu) > Z95 * sigma).mean(), 1 - nuage["part_hors_marge"])

jours = [b for b in eq["boites"] if b["facteur"] == "daysbeforeED" and b["fenetre"] == 30 and b["mesure"] == "optimal_kl"]
assert sum(b["effectif"] for b in jours) == eq["effectifs_fenetres"]["30"], "boîtes incomplètes"

lissage = glissante_equivalents(JOURS_LISSAGE)
lignes = lissage["lignes"]

sortie = {
    "source": "mesure_erreurs (base Jennings et Wlezien, bss.p), via video/donnees.py:TOUS['equivalents'], TOUS['nuage'] et glissante_equivalents",
    "z95": Z95,
    "cible": CIBLE,
    "etapes": ETAPES,
    "facteur_95": facteur_95,
    "nb_lignes": nuage["nb_lignes"],
    "part_hors_marge": nuage["part_hors_marge"],
    # Une entrée par intention de vote : taille, écart au résultat (points), écart-type binomial (points).
    "points": [{"n": float(n[i]), "residu": round(float(residu[i]), 6), "sigma": round(float(sigma[i]), 6)} for i in range(len(pts))],
    "median_reel": eq["median_reel"],
    "medianes": eq["medianes"],
    "nb_proches": eq["nb_proches"],
    "effectif_30_jours": eq["effectifs_fenetres"]["30"],
    "jours": [{k: b[k] for k in ("tranche", "effectif", "q1", "med", "q3")} for b in jours],
    "lissage": {
        "jours_max": JOURS_LISSAGE,
        "effectif": lissage["effectif"],
        "lignes": [{k: round(float(v), 6) for k, v in ligne.items()} for ligne in lignes.to_dict("records")],
    },
}
(ICI.parent / "donnees").mkdir(exist_ok=True)
(ICI.parent / "donnees" / "s4.json").write_text(json.dumps(sortie, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("écrit", ICI.parent / "donnees" / "s4.json", "facteur_95 =", facteur_95)
