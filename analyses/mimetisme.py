"""Consensus d'erreur et resserrement des sondages, élection par élection.

Deux indicateurs, calculés sur les sondages réalisés dans les derniers jours avant le scrutin :

* consensus d'erreur C : les sondages se trompent-ils tous dans le même sens ?
  Pour chaque parti, C = |moyenne des tanh(z / 2)| avec z = (sondage − résultat) / σ ;
  0 : erreurs équilibrées de part et d'autre du résultat, 1 : tous nettement du même côté.
* resserrement R : les sondages sont-ils plus proches les uns des autres que le hasard ne le permet ?
  Pour chaque parti, R = Σ((écart − écart moyen) / σ)² / (k − 1), qui vaut 1 en moyenne pour des
  tirages aléatoires ; R < 1 : sondages trop semblables (mimétisme possible).

Les partis sont agrégés par une moyenne pondérée par leur score. Chaque indicateur est comparé
à 2 000 élections simulées où chaque sondage est un vrai tirage aléatoire (même taille, même
résultat) : le rang indique la part des simulations au moins aussi extrêmes.
"""
import zlib

import numpy as np
import pandas as pd

SEED = 2021
N_SIMULATIONS = 2000
JOURS_MAX = 14
SONDAGES_MIN = 3


def _indicateurs(ecarts, sigma, poids):
    """C et R pour un tableau d'écarts (..., sondages, partis) ; NaN = parti absent du sondage."""
    z = ecarts / sigma
    present = ~np.isnan(z)
    k = present.sum(axis=-2)
    consensus = np.abs(np.nanmean(np.tanh(z / 2), axis=-2))
    w = np.where(present, 1 / sigma**2, 0)
    moy = np.nansum(ecarts * w, axis=-2) / w.sum(axis=-2)
    chi2 = np.nansum(((ecarts - moy[..., None, :]) / sigma) ** 2, axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        resserrement = np.where(k >= SONDAGES_MIN, chi2 / (k - 1), np.nan)
    agrege = lambda v: np.nansum(v * poids, axis=-1) / np.sum(np.where(np.isnan(v), 0, poids), axis=-1)
    return agrege(consensus), agrege(resserrement)


def mesurer_election(s: pd.DataFrame, rng):
    """`s` : lignes (sondage × parti) d'une élection, colonnes idpoll, partyid, n, vote, poll."""
    polls = s.pivot_table(index="idpoll", columns="partyid", values="poll")
    votes = s.groupby("partyid").vote.first().reindex(polls.columns).to_numpy()
    n = s.groupby("idpoll").n.first().reindex(polls.index).to_numpy()
    sigma = np.sqrt(votes * (1 - votes) / n[:, None])
    sigma = np.where(polls.isna(), np.nan, sigma)
    ecarts = polls.to_numpy() - votes
    c_obs, r_obs = _indicateurs(ecarts, sigma, votes)

    # Élections simulées : chaque sondage est un tirage multinomial de taille n, sur le résultat
    # réel complété d'une catégorie « autres » (les parts d'un sondage restent liées).
    p = np.append(votes, max(0.0, 1 - votes.sum()))
    p = p / p.sum()
    tirages = np.stack([rng.multinomial(int(ni), p, size=N_SIMULATIONS)[:, :-1] / ni for ni in n], axis=1)
    ecarts_sim = np.where(polls.isna().to_numpy(), np.nan, tirages - votes)
    c_sim, r_sim = _indicateurs(ecarts_sim, sigma, votes)
    return {
        "consensus": c_obs, "consensus_hasard": np.mean(c_sim), "rang_consensus": np.mean(c_sim >= c_obs),
        "resserrement": r_obs, "rang_resserrement": np.mean(r_sim <= r_obs),
    }


def mimetisme(df: pd.DataFrame):
    """Indicateurs par élection et synthèse. `df` : sortie de analyses.erreurs.get_polls()."""
    s = df[df.daysbeforeED <= JOURS_MAX].copy()
    s["election_id"] = s.country + "|" + s.election + "|" + s.elecdate.astype(str) + "|" + s["round"].astype(int).astype(str)
    lignes = []
    for eid, e in s.groupby("election_id", sort=True):
        if e.idpoll.nunique() < SONDAGES_MIN:
            continue
        # Graine propre à chaque élection : ses résultats ne dépendent pas du périmètre calculé.
        rng = np.random.default_rng([SEED, zlib.crc32(eid.encode())])
        premiere = e.iloc[0]
        lignes.append({
            "pays": premiere.country, "election": premiere.election, "annee": int(premiere.yr),
            "tour": int(premiere["round"]), "sondages": e.idpoll.nunique(),
            "erreur_moyenne": e.erreur.mean(),
            **mesurer_election(e, rng),
        })
    t = pd.DataFrame(lignes)
    return {
        "jours_max": JOURS_MAX,
        "sondages_min": SONDAGES_MIN,
        "nb_elections": len(t),
        "part_consensus": (t.rang_consensus < 0.05).mean() if len(t) else None,
        "part_resserrement": (t.rang_resserrement < 0.05).mean() if len(t) else None,
        "part_les_deux": ((t.rang_consensus < 0.05) & (t.rang_resserrement < 0.05)).mean() if len(t) else None,
        "consensus_median": t.consensus.median() if len(t) else None,
        "consensus_hasard_median": t.consensus_hasard.median() if len(t) else None,
        "resserrement_median": t.resserrement.median() if len(t) else None,
        "elections": t.to_dict("records"),
    }
