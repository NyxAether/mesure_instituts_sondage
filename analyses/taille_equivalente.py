"""Préparation des sondages et simulation des tailles d'échantillon équivalentes.

Reprend le calcul de l'ancien notebook mesure_erreurs/error_plot.ipynb, rendu reproductible :
graine fixe, dérivée pour chaque sondage de son identifiant (le résultat ne dépend donc ni de
l'ordre de calcul ni du nombre de processus).

Étapes :
  1. mesure_erreurs/world_polls.tar.gz -> mesure_erreurs/polls.p  (--preparer, quelques secondes)
  2. mesure_erreurs/polls.p -> mesure_erreurs/bss.p                 (plusieurs heures sur un cœur)

Usage :
  .venv/Scripts/python.exe -m analyses.taille_equivalente --preparer
  .venv/Scripts/python.exe -m analyses.taille_equivalente --jobs 8
  .venv/Scripts/python.exe -m analyses.taille_equivalente --limite 50   # essai -> bss_extrait.p
"""
import argparse
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import kl_div
from tqdm import tqdm

SOURCES = Path(__file__).resolve().parent.parent / "mesure_erreurs"
ARCHIVE = SOURCES / "world_polls.tar.gz"

SEED = 2021
N_TIRAGES = 1000
# Un sondage = un pays, une date (milieu du terrain ; les auteurs moyennent les sondages du même jour) et une élection. Le notebook d'origine omettait
# le pays : deux sondages japonais et norvégien du 15/11/2003 étaient fusionnés.
CLES = ["country", "polldate", "election", "system", "rule", "round", "electionyr", "elecdate"]


def entropie(p, q):
    return -p * np.log(q)


def mse(p, q):
    return np.square(q - p)


def mae(p, q):
    return np.abs(q - p)


MESURES = {"optimal_kl": kl_div, "optimal_entropy": entropie, "optimal_mse": mse, "optimal_mae": mae}


def preparer_sondages(sortie=SOURCES / "polls.p"):
    """Extrait les lignes exploitables de l'archive et numérote les sondages (idpoll)."""
    with tarfile.open(ARCHIVE) as tar:
        # « legacy » : le parseur avec lequel l'ancien polls.p a été produit (valeurs identiques au bit près).
        raw = pd.read_csv(tar.extractfile("worlds_polls.csv"), sep="\t", float_precision="legacy")
    df = raw[raw.vote_.notna() & raw.poll_.notna()].copy()
    df["idpoll"] = df.groupby(CLES, sort=False, dropna=False).ngroup()
    df.to_pickle(sortie)
    return sortie


def distributions(vote, poll):
    """Résultat réel et intentions de vote en proportions, complétés d'une catégorie « autres »."""
    y = vote / 100
    yhat = poll / 100
    if y.sum() < 1:
        y = np.append(y, 1 - y.sum())
        yhat = np.append(yhat, 1 - yhat.sum())
        yhat[yhat < 0] = 0
    yhat[yhat == 0] = 1e-5
    return y, yhat


def mediane_tirages(y, taille, mesure, rng, n_tirages=N_TIRAGES):
    """Mesure médiane entre `y` et `n_tirages` échantillons aléatoires de `taille` personnes."""
    p = y / y.sum()  # certaines élections ont des parts qui ne somment pas exactement à 1
    comptes = rng.multinomial(taille, p, size=n_tirages).astype(float)
    comptes[comptes == 0] = 1e-5
    yhat = comptes / comptes.sum(axis=1, keepdims=True)
    return np.median(mesure(p, yhat).sum(axis=1))


def recherche_taille(reference, y, n, mesure, rng):
    """Plus petite taille dont les tirages aléatoires font (en médiane) au moins aussi mal que `reference`.

    Même recherche que le notebook : doublement tant que le sondage fait mieux que le hasard
    (plafonné à 16 × n), puis dichotomie.
    """
    lb, ub = 1, n
    y_hat = mediane_tirages(y, ub, mesure, rng)
    while reference - y_hat < -1e-5 and ub < 16 * n:
        lb, ub = ub, ub * 2
        y_hat = mediane_tirages(y, ub, mesure, rng)
    if reference - y_hat > 0:
        while ub not in (lb, lb + 1):
            milieu = (ub + lb) // 2
            y_hat = mediane_tirages(y, milieu, mesure, rng)
            if reference - y_hat >= 0:
                ub = milieu
            else:
                lb = milieu
    return ub


def simuler_sondage(tache):
    idpoll, vote, poll, n, meta = tache
    rng = np.random.default_rng([SEED, idpoll])
    y, yhat = distributions(vote, poll)
    ligne = {"id": idpoll, "poll_sample": n}
    for nom, mesure in MESURES.items():
        ligne[nom] = recherche_taille(mesure(y, yhat).sum(), y, n, mesure, rng)
    # Témoin : un vrai tirage aléatoire de même taille, soumis à la même recherche (KL).
    temoin = mediane_tirages(y, n, kl_div, rng, n_tirages=1)
    ligne["oneshot"] = recherche_taille(temoin, y, n, kl_div, rng)
    return {**ligne, **meta}


def taches(polls: pd.DataFrame):
    for idpoll, s in polls.groupby("idpoll", sort=False):
        premiere = s.iloc[0]
        meta = {
            "year": premiere.yr, "country": premiere.country, "election": premiere.election,
            "system": premiere.system, "daysbeforeED": premiere.daysbeforeED,
            "nb_candidates": len(s) + (s.vote_.sum() < 100), "npolls": premiere.npolls,
        }
        yield idpoll, s.vote_.to_numpy(), s.poll_.to_numpy(), int(premiere["sample"]), meta


def simuler(polls: pd.DataFrame, jobs=1):
    liste = list(taches(polls))
    if jobs == 1:
        lignes = [simuler_sondage(t) for t in tqdm(liste)]
    else:
        with ProcessPoolExecutor(jobs) as ex:
            lignes = list(tqdm(ex.map(simuler_sondage, liste, chunksize=16), total=len(liste)))
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preparer", action="store_true", help="régénère polls.p depuis l'archive")
    parser.add_argument("--jobs", type=int, default=1, help="nombre de processus")
    parser.add_argument("--limite", type=int, help="ne simule que les N premiers sondages (essai)")
    parser.add_argument("--sortie", type=Path, help="fichier de sortie (défaut : bss.p, ou bss_extrait.p avec --limite)")
    args = parser.parse_args()

    if args.preparer:
        print(f"Écrit : {preparer_sondages()}")
    else:
        polls = pd.read_pickle(SOURCES / "polls.p")
        polls = polls[polls["sample"] > 0]
        if args.limite:
            polls = polls[polls.idpoll.isin(polls.idpoll.unique()[: args.limite])]
        sortie = args.sortie or SOURCES / ("bss_extrait.p" if args.limite else "bss.p")
        simuler(polls, args.jobs).to_pickle(sortie)
        print(f"Écrit : {sortie}")
