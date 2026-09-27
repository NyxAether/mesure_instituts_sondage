"""Séquence 5 — L'erreur partagée : les sondages d'une même élection se trompent ensemble, sans mimétisme.

Rendu : .venv/Scripts/python.exe -m manim -ql scenes/s5_partage.py Partage
"""
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    Circle,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Transform,
    VGroup,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donnees import TOUS, sondages_election  # noqa: E402
from theme import OPACITE_AIRE, P, SceneRR, entete, fr, libelle, sous_titre, titre  # noqa: E402

NBSP = " "
Z95 = 1.96
SEUIL = 0.05  # une élection est anormale quand moins de 5 % des élections simulées sont aussi extrêmes
EXEMPLE = ("United Kingdom", 2015, 1)
PARTIS = {1: "conservateurs", 2: "travaillistes"}  # identifiants de la base pour le Royaume-Uni


def signe(v):
    return f"+{fr(v)}" if v > 0 else f"−{fr(-v)}"


def essaim(xs, diametre):
    """Hauteurs d'un essaim à une face : chaque point monte jusqu'à ne plus chevaucher les précédents."""
    ordre = np.argsort(xs)
    places, ys = [], np.zeros(len(xs))
    for i in ordre:
        y = 0.0
        while any((xs[i] - px) ** 2 + (y - py) ** 2 < diametre**2 for px, py in places):
            y += diametre / 4
        ys[i] = y
        places.append((xs[i], y))
    return ys


class Partage(SceneRR):
    def construct(self):
        mim = TOUS["mimetisme"]
        elections = mim["elections"]
        ex = sondages_election(*EXEMPLE, mim["jours_max"])
        sondages, resultat = ex["sondages"], ex["resultat"]
        assert resultat.idxmax() == 1 and resultat.drop(1).idxmax() == 2, "partis de l'exemple inattendus"
        fiche = next(e for e in elections if (e["pays"], e["annee"], e["tour"]) == EXEMPLE)
        assert fiche["sondages"] == len(sondages)

        # --- En-tête ----------------------------------------------------------
        tete = entete(5, "l’erreur partagée", "Une erreur ", "partagée").to_corner(UL, buff=0.55)
        source = libelle(
            f"royaume-uni, législatives {EXEMPLE[1]} · {len(sondages)} derniers jours avant le vote, "
            "une moyenne des sondages par jour",
            taille=14,
            couleur=P.discret,
        ).next_to(tete, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(tete, shift=0.15 * DOWN), run_time=1)
        self.play(FadeIn(source), run_time=0.6)

        # --- L'exemple : au hasard, les erreurs se compensent -----------------------
        gauche, droite, bas, haut = -4.8, 3.2, -2.9, 1.2
        y_min, y_max = 27, 41
        j_max = sondages.daysbeforeED.max()

        def pt(jours, score):
            return np.array([
                gauche + (j_max - jours) / (j_max - 1) * (droite - gauche),
                bas + (score - y_min) / (y_max - y_min) * (haut - bas),
                0,
            ])

        graduations = [30, 35, 40]
        grille = VGroup(*[Line(pt(j_max + 0.5, v), pt(0.5, v), color=P.grille, stroke_width=1) for v in graduations])
        lab_y = VGroup(*[libelle(f"{v}{NBSP}%", taille=14).next_to(pt(j_max + 0.5, v), LEFT, buff=0.15) for v in graduations])
        jours_lab = [j_max, 10, 5, 1]
        lab_x = VGroup(*[libelle(fr(j, 0), taille=14).next_to(pt(j, y_min), DOWN, buff=0.15) for j in jours_lab])
        titre_x = libelle("jours avant l’élection", taille=14, couleur=P.discret).next_to(lab_x, DOWN, buff=0.12)
        base = Line(pt(j_max + 0.5, y_min), pt(0.5, y_min), color=P.axe, stroke_width=1.5)

        couleurs = {1: P.series[0], 2: P.series[1]}
        decalage = {1: -0.07, 2: 0.07}
        lignes_resultat, noms = VGroup(), VGroup()
        for parti, nom in PARTIS.items():
            v = resultat[parti]
            ligne = DashedLine(pt(j_max + 0.5, v), pt(0.5, v), color=couleurs[parti], dash_length=0.1, stroke_width=2.5)
            etiquette = VGroup(
                libelle(nom, taille=15, couleur=couleurs[parti]),
                libelle(f"résultat {fr(v)}{NBSP}%", taille=14, couleur=couleurs[parti]),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            etiquette.next_to(pt(0.5, v), RIGHT, buff=0.2).align_to(pt(0.5, v) + UP * 0.05, DOWN if parti == 1 else UP)
            if parti == 2:
                etiquette.shift(DOWN * 0.1)
            lignes_resultat.add(ligne)
            noms.add(etiquette)

        # Sondages simulés : un vrai tirage aléatoire de même taille par jour, sur le résultat réel.
        rng = np.random.default_rng(2015)
        p = np.append(resultat.to_numpy(), 100 - resultat.sum()) / 100
        tirages = np.array([rng.multinomial(int(n), p) / n * 100 for n in sondages["sample"]])
        simules = {parti: tirages[:, list(resultat.index).index(parti)] for parti in PARTIS}
        reels = {parti: sondages[parti].to_numpy() for parti in PARTIS}

        def nuage_sondages(valeurs):
            groupe = VGroup()
            for parti in PARTIS:
                for j, n, v in zip(sondages.daysbeforeED, sondages["sample"], valeurs[parti]):
                    marge = 100 * Z95 * np.sqrt(v / 100 * (1 - v / 100) / n)
                    dx = RIGHT * decalage[parti]
                    trait = Line(pt(j, v - marge) + dx, pt(j, v + marge) + dx).set_stroke(couleurs[parti], 2, opacity=0.45)
                    groupe.add(VGroup(trait, Dot(pt(j, v) + dx, radius=0.055, color=couleurs[parti])))
            return groupe

        def moyennes(valeurs, ecarts=False):
            groupe = VGroup()
            for parti in PARTIS:
                m, v = valeurs[parti].mean(), resultat[parti]
                ligne = Line(pt(j_max + 0.5, m), pt(0.5, m), color=couleurs[parti], stroke_width=5)
                if not ecarts:
                    groupe.add(ligne)
                    continue
                x = RIGHT * 0.2
                bord = VGroup(
                    Line(pt(0.5, m) + x, pt(0.5, v) + x),
                    Line(pt(0.5, m), pt(0.5, m) + x),
                ).set_stroke(couleurs[parti], 2.5)
                texte_ = VGroup(
                    libelle(f"moyenne {fr(m)}{NBSP}%", taille=14, couleur=couleurs[parti]),
                    sous_titre(f"{signe(m - v)} points", taille=24, couleur=couleurs[parti]),
                ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
                texte_.next_to((pt(0.5, m) + pt(0.5, v)) / 2 + x, RIGHT, buff=0.15)
                groupe.add(VGroup(ligne, bord, texte_))
            return groupe

        mode = libelle("si seul le hasard jouait · tirages simulés de même taille", taille=15, couleur=P.texte)
        mode.next_to(pt(j_max + 0.5, y_max), UP, buff=0.05, aligned_edge=LEFT)
        legende = libelle("trait vertical : marge d’erreur de chaque sondage", taille=13, couleur=P.discret)
        legende.next_to(base, UP, buff=0.12).align_to(base, RIGHT)

        points = nuage_sondages(simules)
        moy = moyennes(simules)
        self.play(FadeIn(grille), FadeIn(base), FadeIn(lab_y), FadeIn(lab_x), FadeIn(titre_x), run_time=0.8)
        self.play(*[FadeIn(m) for m in (*lignes_resultat, *noms)], run_time=1)
        self.wait(1)
        self.play(FadeIn(mode), run_time=0.5)
        self.play(FadeIn(points, lag_ratio=0.05), FadeIn(legende), run_time=2)
        self.wait(1)
        self.play(FadeIn(moy), run_time=0.8)
        compense = sous_titre("certains au-dessus, d’autres en dessous : les erreurs se compensent", taille=26)
        compense.next_to(base, DOWN, buff=0.75).align_to(base, LEFT)
        self.play(FadeIn(compense), run_time=0.6)
        self.wait(2.5)

        # --- Les vrais sondages : tous du même côté ----------------------------------
        mode_reel = libelle("les vrais sondages", taille=15, couleur=P.texte).move_to(mode, aligned_edge=LEFT)
        meme_cote = sous_titre("tous du même côté : la moyenne garde l’erreur", taille=26).move_to(compense, aligned_edge=LEFT)
        self.play(FadeOut(moy), FadeOut(compense), Transform(mode, mode_reel), run_time=0.6)
        self.play(Transform(points, nuage_sondages(reels)), run_time=2.5)
        self.wait(1.5)
        moy_reel = moyennes(reels, ecarts=True)
        self.play(FadeIn(moy_reel), run_time=0.8)
        self.play(FadeIn(meme_cote), run_time=0.6)
        self.wait(3)

        exemple = VGroup(grille, base, lab_y, lab_x, titre_x, lignes_resultat, noms, mode, points, legende, moy_reel, meme_cote, source)
        self.play(FadeOut(exemple), run_time=0.8)

        # --- Élection par élection : le consensus d'erreur -------------------------------
        source_m = libelle(
            f"{fr(mim['nb_elections'], 0)} élections depuis 2000 · sondages des {mim['jours_max']} derniers jours, "
            f"au moins {mim['sondages_min']} par élection · un point par élection",
            taille=14,
            couleur=P.discret,
        ).next_to(tete, DOWN, buff=0.2, aligned_edge=LEFT)
        rayon = 0.06
        g_ax, d_ax = -5.6, 3.0
        y_c, y_r = -0.35, -2.95

        def x_c(v):
            return g_ax + v * (d_ax - g_ax)

        def essaim_points(valeurs, y_axe, x_de):
            xs = np.array([x_de(v) for v in valeurs])
            ys = essaim(xs, 2 * rayon * 1.1)
            return [np.array([x, y_axe + rayon * 1.3 + y, 0]) for x, y in zip(xs, ys)]

        axe_c = Line([g_ax, y_c, 0], [d_ax, y_c, 0], color=P.axe, stroke_width=1.5)
        lab_c = VGroup(*[libelle(fr(v, 1 if v == 0.5 else 0), taille=14).next_to([x_c(v), y_c, 0], DOWN, buff=0.12) for v in (0, 0.5, 1)])
        bornes_c = VGroup(
            libelle("erreurs des deux côtés", taille=13, couleur=P.discret).next_to(lab_c[0], DOWN, buff=0.08, aligned_edge=LEFT),
            libelle("toutes du même côté", taille=13, couleur=P.discret).next_to(lab_c[2], DOWN, buff=0.08, aligned_edge=RIGHT),
        )
        nom_c = libelle("consensus d’erreur : les sondages se trompent-ils dans le même sens ?", taille=15, couleur=P.texte)
        nom_c.next_to([g_ax, y_c + 1.75, 0], UP, buff=0, aligned_edge=LEFT)

        # Au hasard, le consensus moyen des élections simulées : une bande étroite, de la plus petite à la plus grande valeur.
        hasard = [e["consensus_hasard"] for e in elections]
        bande = Rectangle(
            width=x_c(max(hasard)) - x_c(min(hasard)), height=0.9,
            stroke_width=0, fill_color=P.texte_2, fill_opacity=OPACITE_AIRE,
        ).move_to([(x_c(min(hasard)) + x_c(max(hasard))) / 2, y_c + 0.45, 0])
        nom_bande = libelle(f"au hasard · autour de {fr(mim['consensus_hasard_median'], 2)}", taille=13, couleur=P.texte_2)
        nom_bande.next_to(bande, UP, buff=0.08, aligned_edge=LEFT)
        pos_reel = essaim_points([e["consensus"] for e in elections], y_c, x_c)
        anormal_c = [e["rang_consensus"] < SEUIL for e in elections]
        dots_c = VGroup(*[Dot(q, radius=rayon, color=P.discret) for q in pos_reel])
        etat_c = libelle(f"les vrais sondages · médiane {fr(mim['consensus_median'], 2)}", taille=14, couleur=P.texte_2)
        etat_c.next_to(nom_c, DOWN, buff=0.12, aligned_edge=LEFT)

        self.play(FadeIn(source_m), FadeIn(axe_c), FadeIn(lab_c), FadeIn(bornes_c), FadeIn(nom_c), run_time=1)
        self.play(FadeIn(bande), FadeIn(nom_bande), run_time=0.8)
        self.wait(1.5)
        self.play(FadeIn(etat_c), FadeIn(dots_c, lag_ratio=0.01), run_time=2)
        self.wait(1)

        def compteur(nom, part, couleur, y_axe):
            return VGroup(
                libelle(nom, taille=15),
                titre("", f"{fr(100 * part, 0)}{NBSP}%", "", taille=56).set_color(couleur),
                libelle(f"attendu au hasard : {fr(100 * SEUIL, 0)}{NBSP}%", taille=15, couleur=P.discret),
            ).arrange(DOWN, aligned_edge=RIGHT, buff=0.08).move_to([6.6, y_axe + 0.2, 0], aligned_edge=DOWN + RIGHT)

        compteur_c = compteur("consensus anormal", mim["part_consensus"], P.series[0], y_c)
        self.play(
            *[d.animate.set_color(P.series[0]) for d, a in zip(dots_c, anormal_c) if a],
            FadeIn(compteur_c),
            run_time=1.5,
        )
        self.wait(1.5)
        i_ex = elections.index(fiche)
        anneau = Circle(radius=rayon * 2.2, color=P.texte, stroke_width=2).move_to(pos_reel[i_ex])
        nom_ex = libelle(f"royaume-uni {EXEMPLE[1]}", taille=13, couleur=P.texte).move_to([pos_reel[i_ex][0], y_c + 1.05, 0])
        fil = Line(nom_ex.get_bottom() + DOWN * 0.05, anneau.get_top(), color=P.texte, stroke_width=1.5)
        self.play(FadeIn(anneau), FadeIn(fil), FadeIn(nom_ex), run_time=0.6)
        self.wait(2)
        self.play(FadeOut(anneau), FadeOut(fil), FadeOut(nom_ex), run_time=0.5)

        # --- Le mimétisme : des sondages trop semblables ? ---------------------------------
        r_min, r_max = 0.1, 100

        def x_r(v):
            return g_ax + (np.log10(v) - np.log10(r_min)) / (np.log10(r_max) - np.log10(r_min)) * (d_ax - g_ax)

        axe_r = Line([g_ax, y_r, 0], [d_ax, y_r, 0], color=P.axe, stroke_width=1.5)
        lab_r = VGroup(*[libelle(fr(v, 1 if v < 1 else 0), taille=14).next_to([x_r(v), y_r, 0], DOWN, buff=0.12) for v in (0.1, 1, 10, 100)])
        repere = DashedLine([x_r(1), y_r, 0], [x_r(1), y_r + 1.1, 0], color=P.discret, dash_length=0.06, stroke_width=1.5)
        bornes_r = VGroup(
            libelle("plus semblables", taille=13, couleur=P.discret).next_to(lab_r[0], DOWN, buff=0.08, aligned_edge=LEFT),
            libelle("= hasard", taille=13, couleur=P.discret).next_to(lab_r[1], DOWN, buff=0.08),
            libelle("plus dispersés", taille=13, couleur=P.discret).next_to(lab_r[3], DOWN, buff=0.08, aligned_edge=RIGHT),
        )
        nom_r = libelle("resserrement : les sondages se ressemblent-ils trop ? (mimétisme)", taille=15, couleur=P.texte)
        nom_r.next_to([g_ax, y_r + 1.25, 0], UP, buff=0, aligned_edge=LEFT)
        pos_r = essaim_points([e["resserrement"] for e in elections], y_r, x_r)
        dots_r = VGroup(*[Dot(q, radius=rayon, color=P.discret) for q in pos_r])
        anormal_r = [e["rang_resserrement"] < SEUIL for e in elections]
        compteur_r = compteur("resserrement anormal", mim["part_resserrement"], P.series[1], y_r)

        self.play(FadeIn(axe_r), FadeIn(lab_r), FadeIn(bornes_r), FadeIn(nom_r), FadeIn(repere), run_time=1)
        self.play(FadeIn(dots_r, lag_ratio=0.01), run_time=1.5)
        self.wait(1)
        self.play(
            *[d.animate.set_color(P.series[1]).scale(1.3) for d, a in zip(dots_r, anormal_r) if a],
            FadeIn(compteur_r),
            run_time=1.5,
        )
        self.wait(3)

        mesures = VGroup(source_m, axe_c, lab_c, bornes_c, nom_c, bande, nom_bande, etat_c, dots_c, compteur_c, axe_r, lab_r, repere, bornes_r, nom_r, dots_r, compteur_r)
        self.play(FadeOut(mesures), run_time=0.8)

        # --- D'où vient-elle ? Ce qu'avancent les instituts, ce que pointent les chercheurs ------
        question = sous_titre("D’où vient cette erreur commune ?", taille=36)
        prudence = libelle("des explications, pas des résultats : l’étude mesure l’erreur, pas son origine", taille=15, couleur=P.discret)

        def colonne(nom, couleur, lignes):
            entete_ = libelle(nom, taille=16, couleur=couleur)
            corps = VGroup(
                *[
                    VGroup(sous_titre(t, taille=26, couleur=P.texte), *([libelle(s, taille=13, couleur=P.discret)] if s else []))
                    .arrange(DOWN, aligned_edge=LEFT, buff=0.06)
                    for t, s in lignes
                ]
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            filet = Line(ORIGIN, DOWN * corps.height, color=couleur, stroke_width=3)
            return VGroup(entete_, VGroup(filet, corps).arrange(RIGHT, buff=0.25, aligned_edge=UP)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        instituts = colonne("ce qu’avancent les instituts", P.series[1], (
            ("l’opinion bouge au dernier moment", "des indécis qui tranchent tard"),
            ("certains électeurs répondent moins", "difficiles à joindre"),
            ("l’abstention est dure à prévoir", None),
        ))
        chercheurs = colonne("ce que pointent les chercheurs", P.series[0], (
            ("tout le monde n’a pas d’opinion sur tout", "Pierre Bourdieu, 1973"),
            ("les questions sont imposées aux sondés", "Pierre Bourdieu, 1973"),
            ("quotas, redressements, formulation", "Alexandre Dézé, 2022"),
        ))
        colonnes = VGroup(instituts, chercheurs).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        VGroup(question, prudence, colonnes).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, -0.6, 0])
        prudence.shift(UP * 0.12)
        commun = sous_titre("dans tous les cas, une erreur que multiplier les sondages ne dilue pas", taille=26, couleur=P.texte_2)
        commun.next_to(colonnes, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(question), FadeIn(prudence), run_time=0.8)
        for col in (instituts, chercheurs):
            self.play(FadeIn(col[0]), FadeIn(col[1][0]), run_time=0.5)
            for ligne in col[1][1]:
                self.play(FadeIn(ligne, shift=0.1 * UP), run_time=0.5)
                self.wait(0.4)
            self.wait(1)
        self.play(FadeIn(commun), run_time=0.6)
        self.wait(2)
        self.play(FadeOut(VGroup(question, prudence, colonnes, commun)), run_time=0.8)

        # --- Le constat ---------------------------------------------------------------------
        constat = titre("Les sondages se trompent ", "ensemble", "", taille=64)
        detail = sous_titre(
            f"consensus d’erreur anormal dans {fr(100 * mim['part_consensus'], 0)}{NBSP}% des élections, "
            f"contre {fr(100 * SEUIL, 0)}{NBSP}% au hasard",
            taille=28,
        )
        moyenne = sous_titre("faire la moyenne des sondages ne corrige pas leur erreur commune", taille=28, couleur=P.texte_2)
        sans = titre("", "Pas", " de mimétisme", taille=52)
        detail_sans = sous_titre(
            f"sondages trop semblables dans {fr(100 * mim['part_resserrement'], 0)}{NBSP}% des élections, "
            f"pas plus qu’au hasard ({fr(100 * SEUIL, 0)}{NBSP}%)",
            taille=28,
        )
        bloc_1 = VGroup(constat, detail, moyenne).arrange(DOWN, buff=0.3)
        bloc_2 = VGroup(sans, detail_sans).arrange(DOWN, buff=0.3)
        VGroup(bloc_1, bloc_2).arrange(DOWN, buff=0.8).move_to([0, -0.5, 0])
        self.play(FadeIn(constat, shift=0.15 * UP), run_time=1)
        self.play(FadeIn(detail), run_time=0.6)
        self.wait(0.8)
        self.play(FadeIn(moyenne), run_time=0.6)
        self.wait(1.5)
        self.play(FadeIn(sans, shift=0.15 * UP), run_time=1)
        self.play(FadeIn(detail_sans), run_time=0.6)
        self.wait(2.5)
