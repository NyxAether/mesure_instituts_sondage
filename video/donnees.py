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


def glissante_equivalents(jours, demi_largeur=0.15, effectif_min=30, bornes=(300, 12_000), points=40):
    """Médiane et quartiles de la taille équivalente (sondages réels et témoin) selon la taille réelle.

    Source : mesure_erreurs/bss.p, filtré comme sur la page (élections depuis annee_min, sondages réalisés au plus
    `jours` jours avant le scrutin). Pour chaque taille x d'une grille logarithmique, on prend les sondages dont la
    taille est à moins de `demi_largeur` décade de x. Contrairement à la médiane glissante de la page (fenêtre de
    rangs après tri), ce lissage ne dépend pas de l'ordre des sondages de même taille.
    """
    import numpy as np
    import pandas as pd

    bss = pd.read_pickle(RACINE / "mesure_erreurs" / "bss.p")
    bss = bss[(bss.year >= TOUS["annee_min"]) & (bss.daysbeforeED <= jours)]
    log_n = np.log10(bss.poll_sample.to_numpy())
    grille = np.geomspace(*bornes, points)
    lignes = []
    for x in grille:
        dans = np.abs(log_n - np.log10(x)) <= demi_largeur
        if dans.sum() < effectif_min:
            continue
        ligne = {"n": x, "effectif": int(dans.sum())}
        for mesure in ("optimal_kl", "oneshot"):
            q1, med, q3 = np.quantile(bss[mesure].to_numpy()[dans], [0.25, 0.5, 0.75])
            ligne |= {f"{mesure}_q1": q1, f"{mesure}_med": med, f"{mesure}_q3": q3}
        lignes.append(ligne)
    return {"effectif": len(bss), "lignes": pd.DataFrame(lignes)}


def sondages_election(pays, annee, tour, jours):
    """Sondages d'une élection (une ligne par sondage, une colonne par parti, en %) et résultat par parti.

    Source : mesure_erreurs/polls.p, filtré comme analyses.erreurs.get_polls (taille connue) et comme le mimétisme
    (sondages réalisés au plus `jours` jours avant le scrutin). La base fusionne parfois les sondages d'un même jour
    en une moyenne : la colonne `sample` est alors la somme des tailles.
    """
    import pandas as pd

    df = pd.read_pickle(RACINE / "mesure_erreurs" / "polls.p")
    s = df[(df.country == pays) & (df.yr == annee) & (df["round"] == tour) & (df.daysbeforeED <= jours) & (df["sample"] > 0)]
    sondages = s.pivot_table(index=["idpoll", "daysbeforeED", "sample"], columns="partyid", values="poll_").reset_index()
    return {"sondages": sondages, "resultat": s.groupby("partyid").vote_.first()}
