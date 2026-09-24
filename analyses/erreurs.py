"""Données de la page « Erreur empirique des sondages » (docs/erreurs.html).

Reprend mesure_erreurs/polls_analysis.ipynb et mesure_erreurs/error_plot.ipynb.
Entrées (non versionnées) : mesure_erreurs/polls.p (sondages et résultats d'élections)
et mesure_erreurs/bss.p (tailles d'échantillon équivalentes, coûteuses à recalculer).

Usage : .venv/Scripts/python.exe -m analyses.erreurs
"""
from pathlib import Path

import numpy as np
import pandas as pd
from loess.loess_2d import loess_2d
from scipy.stats import norm

from analyses.export import write_page_data
from analyses.mimetisme import mimetisme

SOURCES = Path(__file__).resolve().parent.parent / "mesure_erreurs"

Z95 = norm.ppf(0.975)
# Espérance de |N(0, σ)| : c'est elle qu'estime un lissage (moyenne locale) des écarts absolus.
Z_MOYEN = np.sqrt(2 / np.pi)

MESURES = {
    "optimal_kl": "Divergence KL",
    "optimal_entropy": "Entropie croisée",
    "optimal_mse": "Erreur quadratique",
    "optimal_mae": "Erreur absolue",
    "oneshot": "Tirage aléatoire unique",
}


# Fenêtres proposées pour le graphe à boîtes : sondages réalisés au plus N jours avant le
# scrutin (1 semaine, 2 semaines, 1 mois, 3 mois, 1 an). Au-delà, les sondages sont rares.
FENETRES = (7, 14, 30, 90, 365)

# Périmètres proposés par le filtre de la page. Le cas français a peu de sondages :
# moins de tranches et un seuil d'effectif plus bas, pour garder quelque chose à afficher.
PERIMETRES = {
    "tous": {"pays": None, "tranches_taille": 8, "tranches": 5, "tranches_jours": 6, "effectif_min": 30, "glissement": 50, "frac": 0.4},
    "france": {"pays": "France", "tranches_taille": 4, "tranches": 3, "tranches_jours": 3, "effectif_min": 10, "glissement": 20, "frac": 0.7},
}


def sigma(p, n):
    return np.sqrt(p * (1 - p) / n)


def get_polls() -> pd.DataFrame:
    df = pd.read_pickle(SOURCES / "polls.p")
    df = df[df["sample"] > 0].rename(columns={"sample": "n"})
    df = df.astype({"daysbeforeED": int, "yr": int})
    df["vote"] = df.vote_ / 100
    df["poll"] = df.poll_ / 100
    df["residu"] = df.poll - df.vote
    df["erreur"] = df.residu.abs()
    return df


def base(df):
    """Élections depuis 2005, sondages réalisés moins de 8 jours avant le scrutin (milieu du terrain)."""
    sub = df.query("yr >= 2005 and daysbeforeED < 8").copy()
    sub["hors_marge"] = sub.erreur > Z95 * sigma(sub.vote, sub.n)
    return sub


def nuage(sub):
    points = sub[["n", "vote", "poll", "residu", "country", "yr"]].rename(
        columns={"country": "pays", "yr": "annee"}
    )
    ns = np.geomspace(sub.n.min(), sub.n.max(), 80)
    return {
        "points": points.to_dict("records"),
        "marge": [{"n": n, "m": Z95 * sigma(0.5, n)} for n in ns],
        "nb_lignes": len(sub),
        "nb_sondages": sub.idpoll.nunique(),
        "nb_pays": sub.country.nunique(),
        "part_hors_marge": sub.hors_marge.mean(),
        "n_median": sub.n.median(),
    }


def surfaces(sub, frac, nx=40, ny=40):
    """Erreur lissée (LOESS 2D) et erreur théorique moyenne sur une grille (p, log n).

    Matrices indexées [ligne = n][colonne = p], comme l'attend une surface Plotly.
    """
    ps = np.linspace(0.05, 0.6, nx)
    ns = np.geomspace(500, 10_000, ny)
    P, N = np.meshgrid(ps, ns)
    # Axes centrés-réduits à la main : la rotation automatique de loess_2d (rescale=True) échoue
    # quand la plupart des sondages ont la même taille (n = 1 000), comme pour les sondages individuels.
    x, y = sub.vote.values, np.log10(sub.n.values)
    mx, sx, my, sy = x.mean(), x.std(), y.mean(), y.std()
    try:
        Z, _ = loess_2d(
            (x - mx) / sx, (y - my) / sy, sub.erreur.values,
            xnew=(P.ravel() - mx) / sx, ynew=(np.log10(N.ravel()) - my) / sy, degree=1, frac=frac,
        )
    except np.linalg.LinAlgError:
        return None  # trop peu de points distincts pour lisser (ex. France, sondages individuels)
    # Avec peu de points, l'extrapolation linéaire locale peut passer sous zéro.
    Z = np.clip(Z.reshape(P.shape), 0, None)
    th = Z_MOYEN * sigma(P, N)
    ratio = Z / th
    return {"p": ps, "n": ns, "obs": Z, "th": th, "ratio_min": ratio.min(), "ratio_max": ratio.max()}


def par_taille(sub, nb=8):
    """Erreur observée et attendue par tranche de taille d'échantillon (quantiles)."""
    sub = sub.assign(
        attendue=Z_MOYEN * sigma(sub.vote, sub.n),
        tranche=pd.qcut(sub.n, nb, duplicates="drop"),
    )
    rows = []
    for _, g in sub.groupby("tranche", observed=True):
        obs, att = g.erreur.mean(), g.attendue.mean()
        pq = (g.vote * (1 - g.vote)).mean()
        rows.append({
            "n_min": g.n.min(), "n_max": g.n.max(), "n": g.n.median(), "lignes": len(g),
            "obs": obs, "th": att,
            # Taille d'un sondage aléatoire qui aurait la même erreur moyenne.
            "n_equiv": Z_MOYEN**2 * pq / obs**2,
            "hors_marge": g.hors_marge.mean(),
        })
    return rows


def boites(df, colonne, tranches, effectif_min, mesures=MESURES):
    """Statistiques de boîtes à moustaches (1,5 × IQR, sans points extrêmes) par tranche."""
    rows = []
    for _, g in df.groupby(tranches, observed=True):
        if len(g) < effectif_min:
            continue
        lo, hi = g[colonne].min(), g[colonne].max()
        label = f"{lo:.0f}" if lo == hi else f"{lo:.0f}–{hi:.0f}"
        for m in mesures:
            v = g[m]
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            rows.append({
                "facteur": colonne, "tranche": label, "mesure": m, "effectif": len(v),
                "q1": q1, "med": med, "q3": q3,
                "bas": v[v >= q1 - 1.5 * iqr].min(), "haut": v[v <= q3 + 1.5 * iqr].max(),
            })
    return rows


def equivalents(bss, cfg):
    proches = bss[bss.daysbeforeED <= 14]
    nb, mini = cfg["tranches"], cfg["effectif_min"]
    # Boîtes précalculées pour chaque fenêtre avant le scrutin (la simulation
    # coûteuse est déjà dans bss.p : filtrer et regrouper ne prend qu'une fraction de seconde).
    facteurs = []
    for fenetre in FENETRES:
        f = bss[bss.daysbeforeED <= fenetre]
        rows = (
            boites(f, "poll_sample", pd.qcut(f.poll_sample, nb, duplicates="drop"), mini)
            + boites(f, "year", pd.qcut(f.year, nb, duplicates="drop"), mini)
            + boites(f, "daysbeforeED", pd.qcut(f.daysbeforeED, cfg["tranches_jours"], duplicates="drop"), mini)
            + boites(f, "nb_candidates", f.nb_candidates, mini)
        )
        facteurs += [{**r, "fenetre": fenetre} for r in rows]
    # Médiane glissante de la taille équivalente selon la taille réelle.
    k = cfg["glissement"]
    tri = proches.sort_values("poll_sample")
    glissante = tri[["poll_sample", *MESURES]].rolling(k, center=True).median().dropna()
    glissante = glissante[glissante.poll_sample <= 6000].iloc[:: max(1, k // 5)]
    return {
        "mesures": MESURES,
        "glissement": k,
        "effectif_min": mini,
        "nb_sondages": len(bss),
        "nb_proches": len(proches),
        "effectifs_fenetres": {str(w): int((bss.daysbeforeED <= w).sum()) for w in FENETRES},
        "median_reel": proches.poll_sample.median(),
        "medianes": {m: proches[m].median() for m in MESURES},
        "glissante": glissante.to_dict("records"),
        "boites": facteurs,
    }


def perimetre(df, bss, cfg):
    if cfg["pays"]:
        df = df[df.country == cfg["pays"]]
        bss = bss[bss.country == cfg["pays"]]
    sub = base(df)
    return {
        "source": {
            "lignes": len(df), "sondages": df.idpoll.nunique(), "pays": df.country.nunique(),
            "debut": df.yr.min(), "fin": df.yr.max(),
        },
        "nuage": nuage(sub),
        "surfaces": surfaces(sub, cfg["frac"]),
        "par_taille": par_taille(sub, cfg["tranches_taille"]),
        "equivalents": equivalents(bss, cfg),
        "mimetisme": mimetisme(df),
    }


def build(individuels=False):
    """`individuels` : écarte les lignes où les auteurs de la base ont moyenné plusieurs sondages du même jour."""
    df = get_polls()
    bss = pd.read_pickle(SOURCES / "bss.p")
    if individuels:
        df = df[df.npolls == 1]
        bss = bss[bss.id.isin(df.idpoll)]
    return {nom: perimetre(df, bss, cfg) for nom, cfg in PERIMETRES.items()}


def ecrire_page_individuels():
    """Variante personnelle de docs/erreurs.html (non liée depuis l'index), générée pour rester synchronisée."""
    docs = Path(__file__).resolve().parent.parent / "docs"
    html = (docs / "erreurs.html").read_text(encoding="utf-8")
    remplacements = {
        "<title>Erreur empirique des sondages</title>": "<title>Erreur empirique, sondages individuels</title>",
        "<body>": '<body data-donnees="erreurs_individuels">',
        '<script src="data/erreurs.js"></script>': '<script src="data/erreurs_individuels.js"></script>',
        '    <p class="lede">': (
            '    <p class="note warning"><strong>Variante de travail.</strong> Mêmes calculs que la page principale, '
            "mais sans les lignes où les auteurs de la base ont moyenné plusieurs sondages du même jour "
            "(<code>npolls</code> &gt; 1). Les chiffres affichés sont recalculés ; les commentaires rédigés, eux, "
            "décrivent la page principale.</p>\n"
            '    <p class="lede">'
        ),
    }
    for avant, apres in remplacements.items():
        assert html.count(avant) == 1, avant
        html = html.replace(avant, apres)
    sortie = docs / "erreurs_individuels.html"
    sortie.write_text("<!-- Fichier généré par analyses/erreurs.py --individuels : ne pas modifier à la main. -->\n" + html, encoding="utf-8", newline="\n")
    return sortie


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--individuels", action="store_true", help="variante sans les moyennes journalières (npolls > 1)")
    args = parser.parse_args()
    if args.individuels:
        print(f"Écrit : {write_page_data('erreurs_individuels', build(individuels=True))}")
        print(f"Écrit : {ecrire_page_individuels()}")
    else:
        print(f"Écrit : {write_page_data('erreurs', build())}")
