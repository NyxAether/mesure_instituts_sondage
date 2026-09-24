"""Données de la page « Marges d'erreur et variations » (docs/explications.html).

Usage : .venv/Scripts/python.exe -m analyses.explications
"""
import numpy as np
from scipy.stats import norm

from analyses.export import write_page_data

SEED = 2021


def p_inversion(m1, std1, m2, std2):
    """Probabilité que le tirage de N(m1, std1) dépasse celui de N(m2, std2)."""
    return norm.cdf((m1 - m2) / np.sqrt(std1**2 + std2**2))


def sample_norm(rng, mean1=0, std1=1, mean2=0, std2=1, size=1_000_000):
    """Estimation Monte-Carlo de la même probabilité."""
    a = rng.normal(mean1, std1, size)
    b = rng.normal(mean2, std2, size)
    return np.sum(a > b) / size


def inversion_vs_distance(rng):
    """Probabilité d'inversion selon la distance (en σ) entre deux valeurs observées."""
    distances = np.round(np.linspace(0, 5, 101), 3)
    courbe = [{"d": d, "p": p_inversion(0, 1, d, 1)} for d in distances]
    monte_carlo = [
        {"d": d, "p": sample_norm(rng, 0, 1, d, 1, size=200_000)}
        for d in np.arange(0.5, 5.01, 0.5)
    ]
    seuil_analytique = np.sqrt(2) * norm.ppf(0.95)
    return {
        "courbe": courbe,
        "monte_carlo": monte_carlo,
        "seuil": 2.328,
        "p_seuil_mc": sample_norm(rng, 0, 1, 2.328, 1),
        "seuil_analytique": seuil_analytique,
    }


def pile_ou_face(rng, lancers=100, repetitions=1000):
    """Somme de 100 lancers (pile = 1, face = -1), répétée 1000 fois."""
    sommes = rng.choice([-1, 1], (lancers, repetitions)).sum(axis=0)
    valeurs, effectifs = np.unique(sommes, return_counts=True)
    return {
        "lancers": lancers,
        "repetitions": repetitions,
        "effectifs": [{"somme": v, "n": n} for v, n in zip(valeurs, effectifs)],
    }


def echantillons_femmes(rng, taille=1000, repetitions=1000, p=0.54):
    """Nombre de femmes dans 1000 échantillons de 1000 personnes."""
    nb = rng.choice([0, 1], (taille, repetitions), p=[1 - p, p]).sum(axis=0)
    moyenne = nb.mean()
    ecart_type = nb.std(ddof=1)
    valeurs, effectifs = np.unique(nb, return_counts=True)
    return {
        "taille": taille,
        "repetitions": repetitions,
        "p": p,
        "moyenne": moyenne,
        "ecart_type": ecart_type,
        "min": nb.min(),
        "max": nb.max(),
        "part_dans_ecart_type": np.mean(np.abs(nb - moyenne) <= ecart_type),
        "effectifs": [{"femmes": v, "n": n} for v, n in zip(valeurs, effectifs)],
    }


def recherche_coeff(coeff, m, n, n2):
    """Probabilité d'inversion quand chaque valeur est à `coeff` écarts-types de la frontière.

    L'écart-type de la seconde valeur dépend de sa propre position, d'où le point fixe.
    """
    std = np.sqrt(m * (1 - m) / n)
    std2 = np.sqrt(m * (1 - m) / n2)
    for _ in range(1000):
        m2 = m + std * coeff + std2 * coeff
        std2 = np.sqrt(m2 * (1 - m2) / n2)
    return p_inversion(m, std, m2, std2)


def correction_taille():
    coeffs = np.round(np.linspace(0.8, 1.8, 101), 3)
    selon_n = [
        {"serie": f"n2 = {k} × n1" if k > 1 else "n2 = n1", "coeff": c,
         "p": 2 * recherche_coeff(c, 0.5, 1000, 1000 * k)}
        for k in [1, 2, 3, 4]
        for c in coeffs
    ]
    selon_p = [
        {"serie": f"p = {p}", "coeff": c, "p": 2 * recherche_coeff(c, p, 1000, 1000)}
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]
        for c in coeffs
    ]
    return {"selon_n": selon_n, "selon_p": selon_p}


def build():
    rng = np.random.default_rng(SEED)
    return {
        "inversion": inversion_vs_distance(rng),
        "pile_ou_face": pile_ou_face(rng),
        "femmes": echantillons_femmes(rng),
        "correction": correction_taille(),
    }


if __name__ == "__main__":
    print(f"Écrit : {write_page_data('explications', build())}")
