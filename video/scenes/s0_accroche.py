"""Séquence 0 — Accroche : primaire de la droite 2016, puis second tour de la présidentielle 2017.

Rendu : .venv/Scripts/python.exe -m manim -ql scenes/s0_accroche.py Accroche
"""
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    VGroup,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donnees import TOUS, primaire_2016, sondages_election  # noqa: E402
from theme import P, SceneRR, fr, libelle, sous_titre, titre  # noqa: E402

NBSP = " "
PRESIDENTIELLE = ("France", 2017, 2)
CANDIDATS_2017 = {13: "Macron", 3: "Le Pen"}  # identifiants de la base


def signe(v):
    return f"+{fr(v)}" if v > 0 else f"−{fr(-v)}"


def pour_cent(v):
    return f"{fr(v, 0 if float(v).is_integer() else 1)}{NBSP}%"


class Accroche(SceneRR):
    def construct(self):
        prim = primaire_2016()
        sondages, resultat = prim["sondages"], prim["resultat"]
        candidats = {"fillon": ("Fillon", P.series[0]), "juppe": ("Juppé", P.series[1]), "sarkozy": ("Sarkozy", P.muet)}

        # --- Primaire de la droite 2016 --------------------------------------------------
        tag = libelle("primaire de la droite et du centre · 1er tour · novembre 2016", taille=16, couleur=P.discret)
        tag.to_corner(UL, buff=0.55)
        gauche, droite, bas, haut = -5.2, 2.6, -2.6, 2.3
        x_res = 4.6
        y_max = 50

        def y(v):
            return bas + v / y_max * (haut - bas)

        def x(i):
            return gauche + i / (len(sondages) - 1) * (droite - gauche)

        graduations = [0, 10, 20, 30, 40, 50]
        grille = VGroup(*[Line([gauche - 0.3, y(v), 0], [x_res + 0.4, y(v), 0], color=P.grille, stroke_width=1) for v in graduations])
        lab_y = VGroup(*[libelle(f"{v}{NBSP}%", taille=13).next_to([gauche - 0.3, y(v), 0], LEFT, buff=0.12) for v in graduations])
        lab_x = VGroup(*[
            VGroup(libelle(s["institut"].split()[0].lower(), taille=11), libelle(s["fin"][-2:] + " nov.", taille=11, couleur=P.discret))
            .arrange(DOWN, buff=0.04).next_to([x(i), bas, 0], DOWN, buff=0.12)
            for i, s in enumerate(sondages)
        ])
        lab_res = libelle("résultat 20 nov.", taille=12, couleur=P.texte).next_to([x_res, bas, 0], DOWN, buff=0.12)
        source = libelle("sondages publiés après le deuxième débat · source : wikipédia", taille=12, couleur=P.discret)
        source.next_to(lab_x, DOWN, buff=0.15).align_to(lab_x, LEFT)

        courbes, noms = VGroup(), VGroup()
        for cle, (nom, couleur) in candidats.items():
            points = [np.array([x(i), y(s[cle]), 0]) for i, s in enumerate(sondages)]
            ligne = VGroup(*[Line(a, b, color=couleur, stroke_width=2.5) for a, b in zip(points, points[1:])])
            dots = VGroup(*[Dot(q, radius=0.06, color=couleur) for q in points])
            courbes.add(VGroup(ligne, dots))
            noms.add(libelle(nom.lower(), taille=14, couleur=couleur).next_to(points[0], DOWN if cle == "fillon" else UP, buff=0.15))

        self.play(FadeIn(tag), FadeIn(grille), FadeIn(lab_y), run_time=0.8)
        self.play(FadeIn(lab_x, lag_ratio=0.1), FadeIn(source), run_time=1)
        for (ligne, dots), nom in zip(courbes, noms):
            self.play(FadeIn(nom), FadeIn(dots, lag_ratio=0.1), Create(ligne), run_time=1.2)
        premier = sondages[0]
        fillon_debut = sous_titre(f"Fillon troisième, {pour_cent(premier['fillon'])}", taille=26, couleur=P.series[0])
        fillon_debut.next_to(noms[0], DOWN, buff=0.2).align_to([gauche, 0, 0], LEFT)
        self.play(FadeIn(fillon_debut), run_time=0.6)
        self.wait(1.5)

        # Le résultat du premier tour
        dernier = len(sondages) - 1
        sauts, points_res, lab_valeurs, ecarts = VGroup(), VGroup(), VGroup(), VGroup()
        for cle, (nom, couleur) in candidats.items():
            depart = np.array([x(dernier), y(sondages[dernier][cle]), 0])
            arrivee = np.array([x_res, y(resultat[cle]), 0])
            sauts.add(DashedLine(depart, arrivee, color=couleur, dash_length=0.08, stroke_width=2))
            points_res.add(Dot(arrivee, radius=0.11, color=couleur))
            lab_valeurs.add(libelle(pour_cent(resultat[cle]), taille=15, couleur=couleur).next_to(arrivee, RIGHT, buff=0.15))
            ecart = resultat[cle] - sondages[dernier][cle]
            ecarts.add(libelle(f"{signe(ecart)} pts", taille=13, couleur=couleur).next_to(lab_valeurs[-1], DOWN, buff=0.06, aligned_edge=LEFT))
        self.play(FadeIn(lab_res), *[Create(s) for s in sauts], run_time=1.2)
        self.play(FadeIn(points_res, scale=1.5), FadeIn(lab_valeurs), run_time=0.8)
        fillon_fin = sous_titre(f"le soir du vote : {pour_cent(resultat['fillon'])}", taille=30, couleur=P.series[0])
        fillon_fin.next_to(points_res[0], UP, buff=0.35).align_to(points_res[0], RIGHT).shift(RIGHT * 0.3)
        self.play(FadeIn(fillon_fin, shift=0.1 * UP), run_time=0.6)
        self.wait(1)
        nom_ecart = libelle(
            f"+/− : écart au dernier sondage ({sondages[dernier]['institut'].lower()}, {sondages[dernier]['terrain']})",
            taille=12, couleur=P.discret,
        )
        nom_ecart.next_to(source, DOWN, buff=0.08).align_to(lab_res, RIGHT)
        self.play(FadeIn(ecarts, lag_ratio=0.2), FadeIn(nom_ecart), run_time=1)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)

        # --- Accident isolé ? Présidentielle 2017, second tour ------------------------------------
        accident = sous_titre("Accident isolé ?", taille=40)
        self.play(FadeIn(accident), run_time=0.6)
        self.wait(1)
        self.play(FadeOut(accident), run_time=0.5)

        mim = TOUS["mimetisme"]
        ex = sondages_election(*PRESIDENTIELLE, mim["jours_max"])
        points_2017, res_2017 = ex["sondages"], ex["resultat"]
        assert set(res_2017.index) == set(CANDIDATS_2017) and res_2017.idxmax() == 13, "candidats de 2017 inattendus"
        fiche = next(e for e in mim["elections"] if (e["pays"], e["annee"], e["tour"]) == PRESIDENTIELLE)
        assert fiche["sondages"] == len(points_2017)

        tag_2 = libelle(
            f"présidentielle 2017 · 2d tour · {mim['jours_max']} derniers jours, une moyenne des sondages par jour",
            taille=16, couleur=P.discret,
        ).to_corner(UL, buff=0.55)
        echelle = libelle("chaque bande : 6 points de part et d’autre du résultat", taille=12, couleur=P.discret)
        echelle.next_to(tag_2, DOWN, buff=0.15, aligned_edge=LEFT)
        j_max = points_2017.daysbeforeED.max()
        couleurs = {13: P.series[1], 3: P.series[0]}
        # Deux panneaux, un par candidat, chacun centré sur le résultat (± 6 points).
        panneaux = {13: (0.9, 2.6), 3: (-2.4, -0.7)}  # (bas, haut) de chaque panneau

        def pt(cand, jours, v):
            b, h = panneaux[cand]
            r = res_2017[cand]
            return np.array([-5.2 + (j_max - jours) / (j_max - 1) * 8.4, b + (v - (r - 6)) / 12 * (h - b), 0])

        elements, dots_2017 = VGroup(), VGroup()
        for cand, nom in CANDIDATS_2017.items():
            r = res_2017[cand]
            ligne = DashedLine(pt(cand, j_max + 0.5, r), pt(cand, 0.5, r), color=couleurs[cand], dash_length=0.1, stroke_width=2.5)
            etiquette = VGroup(
                libelle(nom.lower(), taille=15, couleur=couleurs[cand]),
                libelle(f"résultat {pour_cent(round(r, 1))}", taille=13, couleur=couleurs[cand]),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.05).next_to(pt(cand, 0.5, r), RIGHT, buff=0.2)
            elements.add(ligne, etiquette)
            dots_2017.add(VGroup(*[
                Dot(pt(cand, j, v), radius=0.07, color=couleurs[cand]) for j, v in zip(points_2017.daysbeforeED, points_2017[cand])
            ]))
        lab_j = VGroup(*[libelle(fr(j, 0), taille=12).next_to(pt(3, j, res_2017[3] - 6), DOWN, buff=0.1) for j in (j_max, 10, 5, 2)])
        titre_j = libelle("jours avant l’élection", taille=12, couleur=P.discret).next_to(lab_j, DOWN, buff=0.08)

        self.play(FadeIn(tag_2), FadeIn(elements), FadeIn(lab_j), FadeIn(titre_j), run_time=0.8)
        self.play(FadeIn(echelle), run_time=0.4)
        for groupe in dots_2017:
            self.play(FadeIn(groupe, lag_ratio=0.1), run_time=1)
        dessous = all(v < res_2017[13] for v in points_2017[13])
        dessus = all(v > res_2017[3] for v in points_2017[3])
        assert dessous and dessus, "l'exemple 2017 n'est plus tout d'un côté"
        constat = sous_titre("tous du même côté du résultat : Macron sous-estimé, Le Pen surestimée", taille=28).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(constat), run_time=0.6)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)

        # --- Titre ------------------------------------------------------------------------------
        question = titre("Que valent ", "vraiment", " les sondages ?", taille=60)
        base = libelle(
            f"{fr(mim['nb_elections'], 0)} élections · {TOUS['selection']['pays']} pays · "
            f"{fr(TOUS['selection']['sondages'], 0)} sondages confrontés aux résultats",
            taille=17, couleur=P.texte_2,
        )
        VGroup(question, base).arrange(DOWN, buff=0.45)
        self.play(FadeIn(question, shift=0.15 * UP), run_time=1.2)
        self.play(FadeIn(base), run_time=0.8)
        self.wait(2.5)
