"""Exporte en JSON les chiffres de la séquence 3 (taille de l'échantillon), lus par video/donnees.py comme dans la version Manim.

Lancement (depuis la racine du dépôt) : video/.venv/Scripts/python.exe video-js/export/exporter_s3.py
Écrit video-js/donnees/s3.json.
"""
import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent.parent / "video"))
from donnees import TOUS  # noqa: E402

nuage = TOUS["nuage"]
sortie = {
    "source": "video/donnees.py : nuage (écarts sondage − résultat de la séquence 2) et par_taille (7 tranches de taille)",
    "nb_lignes": nuage["nb_lignes"],
    "points": [{"n": p["n"], "residu": p["residu"]} for p in nuage["points"]],
    "par_taille": [
        {champ: t[champ] for champ in ("n_min", "n_max", "n", "lignes", "obs", "th")} for t in TOUS["par_taille"]
    ],
}
assert len(sortie["points"]) == nuage["nb_lignes"], "le nombre de points ne correspond plus à nb_lignes"
(ICI.parent / "donnees").mkdir(exist_ok=True)
(ICI.parent / "donnees" / "s3.json").write_text(json.dumps(sortie, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
print("écrit", ICI.parent / "donnees" / "s3.json")
