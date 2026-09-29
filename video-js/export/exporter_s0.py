"""Exporte en JSON les chiffres de la séquence 0 (accroche), lus par video/donnees.py comme dans la version Manim.

Lancement (depuis la racine du dépôt) : video/.venv/Scripts/python.exe video-js/export/exporter_s0.py
Écrit video-js/donnees/s0.json.
"""
import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent.parent / "video"))
from donnees import TOUS, primaire_2016, sondages_election  # noqa: E402

PRESIDENTIELLE = ("France", 2017, 2)
CANDIDATS_2017 = {13: "macron", 3: "lepen"}  # identifiants de la base

mim = TOUS["mimetisme"]
ex = sondages_election(*PRESIDENTIELLE, mim["jours_max"])
sondages, resultat = ex["sondages"], ex["resultat"]
fiche = next(e for e in mim["elections"] if (e["pays"], e["annee"], e["tour"]) == PRESIDENTIELLE)
assert fiche["sondages"] == len(sondages), "le nombre de moyennes quotidiennes ne correspond plus à la page"

sortie = {
    "primaire_2016": primaire_2016(),
    "presidentielle_2017": {
        "source": "mesure_erreurs/polls.p (base Jennings et Wlezien), filtré comme video/donnees.py:sondages_election",
        "jours_max": mim["jours_max"],
        "note": "une ligne par jour : la base fusionne les sondages d'un même jour en une moyenne",
        "sondages": [
            {"jours_avant": int(ligne.daysbeforeED), **{nom: round(float(ligne.loc[cid]), 4) for cid, nom in CANDIDATS_2017.items()}}
            for _, ligne in sondages.sort_values("daysbeforeED", ascending=False).iterrows()
        ],
        "resultat": {nom: round(float(resultat[cid]), 2) for cid, nom in CANDIDATS_2017.items()},
    },
    "titre": {
        "elections": mim["nb_elections"],
        "pays": TOUS["selection"]["pays"],
        "sondages": TOUS["selection"]["sondages"],
    },
}
(ICI.parent / "donnees").mkdir(exist_ok=True)
(ICI.parent / "donnees" / "s0.json").write_text(json.dumps(sortie, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("écrit", ICI.parent / "donnees" / "s0.json")
