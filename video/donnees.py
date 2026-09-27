"""Chargement des données exportées pour la page docs/erreurs.html (source unique des chiffres de la vidéo)."""
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]


def charger(page="erreurs"):
    """Renvoie le dictionnaire exporté par analyses/ pour une page (clés « tous » et « france »)."""
    source = (RACINE / "docs" / "data" / f"{page}.js").read_text(encoding="utf-8")
    prefixe = f'window.DATA["{page}"] = '
    debut = source.index(prefixe) + len(prefixe)
    return json.loads(source[debut:].rstrip().rstrip(";"))


ERREURS = charger("erreurs")
TOUS = ERREURS["tous"]
FRANCE = ERREURS["france"]
