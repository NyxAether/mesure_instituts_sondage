"""Exporte en JSON les chiffres de la séquence 5 (erreur partagée), lus par video/donnees.py comme dans la version Manim.

Lancement (depuis la racine du dépôt) : video/.venv/Scripts/python.exe video-js/export/exporter_s5.py
Écrit video-js/donnees/s5.json.

Les sondages « si seul le hasard jouait » sont le même tirage multinomial que dans la scène Manim d'origine (graine 2015).
"""
import json
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent.parent / "video"))
from donnees import TOUS, sondages_election  # noqa: E402

Z95 = 1.96
SEUIL = 0.05  # une élection est anormale quand moins de 5 % des élections simulées sont aussi extrêmes
GRAINE = 2015
EXEMPLE = ("United Kingdom", 2015, 1)
PARTIS = {1: "conservateurs", 2: "travaillistes"}  # identifiants de la base pour le Royaume-Uni

mim = TOUS["mimetisme"]
elections = mim["elections"]
ex = sondages_election(*EXEMPLE, mim["jours_max"])
sondages, resultat = ex["sondages"], ex["resultat"]
assert resultat.idxmax() == 1 and resultat.drop(1).idxmax() == 2, "partis de l'exemple inattendus"
fiche = next(e for e in elections if (e["pays"], e["annee"], e["tour"]) == EXEMPLE)
assert fiche["sondages"] == len(sondages)

# Sondages simulés : un vrai tirage aléatoire de même taille par jour, sur le résultat réel.
rng = np.random.default_rng(GRAINE)
p = np.append(resultat.to_numpy(), 100 - resultat.sum()) / 100
tirages = np.array([rng.multinomial(int(n), p) / n * 100 for n in sondages["sample"]])
simules = {parti: tirages[:, list(resultat.index).index(parti)] for parti in PARTIS}

sortie = {
    "source": "mesure_erreurs/polls.p (base Jennings et Wlezien) et docs/data/erreurs.js, via video/donnees.py",
    "z95": Z95,
    "seuil": SEUIL,
    "graine": GRAINE,
    "exemple": {"pays": EXEMPLE[0], "annee": EXEMPLE[1], "tour": EXEMPLE[2], "index": elections.index(fiche)},
    "partis": {nom: parti for parti, nom in PARTIS.items()},
    "sondages": [
        {
            "jours_avant": int(ligne.daysbeforeED),
            "taille": int(ligne["sample"]),
            **{nom: round(float(ligne.loc[parti]), 6) for parti, nom in PARTIS.items()},
        }
        for _, ligne in sondages.iterrows()
    ],
    "resultat": {nom: round(float(resultat[parti]), 4) for parti, nom in PARTIS.items()},
    "simules": {nom: [round(float(v), 6) for v in simules[parti]] for parti, nom in PARTIS.items()},
    "mimetisme": {k: v for k, v in mim.items() if k != "elections"},
    "elections": [
        {k: e[k] for k in ("pays", "annee", "tour", "sondages", "consensus", "consensus_hasard", "rang_consensus", "resserrement", "rang_resserrement")}
        for e in elections
    ],
}
(ICI.parent / "donnees").mkdir(exist_ok=True)
(ICI.parent / "donnees" / "s5.json").write_text(json.dumps(sortie, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("écrit", ICI.parent / "donnees" / "s5.json")
