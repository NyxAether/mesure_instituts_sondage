"""Exporte en JSON les chiffres de la séquence 6 (présage), lus par video/donnees.py comme dans la version Manim.

Lancement (depuis la racine du dépôt) : video/.venv/Scripts/python.exe video-js/export/exporter_s6.py
Écrit video-js/donnees/s6.json.
"""
import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent.parent / "video"))
from donnees import FRANCE, TOUS  # noqa: E402

selection = TOUS["selection"]
sortie = {
    "source": "video/donnees.py (mesure_erreurs), mêmes chiffres que la scène Manim s6_presage.py",
    "base": {
        "fin": TOUS["source"]["fin"],
        "annee_min": TOUS["annee_min"],
        "sondages": selection["sondages"],
        "pays": selection["pays"],
    },
    "france": {"sondages": FRANCE["selection"]["sondages"]},
    # Par tranche de taille : effectif n, erreur typique attendue au hasard (th) et observée (obs), en points.
    "par_taille": [{"n": t["n"], "th": t["th"], "obs": t["obs"]} for t in TOUS["par_taille"]],
}
(ICI.parent / "donnees").mkdir(exist_ok=True)
(ICI.parent / "donnees" / "s6.json").write_text(json.dumps(sortie, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("écrit", ICI.parent / "donnees" / "s6.json")
