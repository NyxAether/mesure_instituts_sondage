"""Séquence 4 — Taille équivalente : on élargit l'entonnoir jusqu'à contenir 95 % des écarts.

Rendu : .venv/Scripts/python.exe -m manim -ql scenes/s4_equivalente.py Equivalente
"""
import math
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    Axes,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Polygon,
    Rectangle,
    VGroup,
    VMobject,
    ValueTracker,
    always_redraw,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donnees import TOUS, glissante_equivalents  # noqa: E402
from theme import OPACITE_AIRE, P, SceneRR, entete, fr, libelle, sous_titre, titre  # noqa: E402

NBSP = " "
Z95 = 1.96
CIBLE = 0.95  # part des écarts que l'entonnoir élargi doit contenir
Y_MAX = 14  # points ; les écarts exportés sont bornés à ± 14
GRADUATIONS_N = [300, 1_000, 3_000, 10_000, 30_000, 100_000]
ETAPES = [2, 4]  # facteurs montrés avant le facteur final


def entier(x):
    """Arrondi à l'entier le plus proche (0,5 vers le haut), comme la page."""
    return int(math.floor(x + 0.5))


def signe(v):
    return f"+{fr(v, 0)}" if v > 0 else ("0" if v == 0 else f"−{fr(-v, 0)}")


class Equivalente(SceneRR):
    def construct(self):
        eq = TOUS["equivalents"]
        med = eq["medianes"]
        n_reel, n_kl, n_temoin = eq["median_reel"], med["optimal_kl"], med["oneshot"]
        jours = [b for b in eq["boites"] if b["facteur"] == "daysbeforeED" and b["fenetre"] == 30 and b["mesure"] == "optimal_kl"]
        assert sum(b["effectif"] for b in jours) == eq["effectifs_fenetres"]["30"], "boîtes incomplètes"

        nuage = TOUS["nuage"]
        pts = nuage["points"]
        n = np.array([p["n"] for p in pts])
        vote = np.array([p["vote"] for p in pts])
        residu = 100 * np.array([p["residu"] for p in pts])
        sigma = 100 * np.sqrt(vote * (1 - vote) / n)
        # Facteur de division des tailles pour lequel 95 % des écarts tiennent dans leur marge (marge propre,
        # comme le 45 % de la séquence 2). Les points, eux, sont colorés selon l'entonnoir tracé (p = 50 %).
        facteur_95 = np.quantile((np.abs(residu) / (Z95 * sigma)) ** 2, CIBLE)
        assert np.isclose(1 - (np.abs(residu) > Z95 * sigma).mean(), 1 - nuage["part_hors_marge"])

        def dedans(f):
            return (np.abs(residu) <= Z95 * sigma * np.sqrt(f)).mean()

        # --- En-tête ----------------------------------------------------------
        tete = entete(4, "la taille équivalente", "Combien vaut ", "vraiment", f"{NBSP}un sondage{NBSP}?").to_corner(UL, buff=0.55)
        source = libelle(
            f"les mêmes {fr(nuage['nb_lignes'], 0)} intentions de vote qu’à la séquence 2 · dernière semaine avant le vote",
            taille=14,
            couleur=P.discret,
        ).next_to(tete, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(tete, shift=0.15 * DOWN), run_time=1)
        self.play(FadeIn(source), run_time=0.6)

        # --- L'entonnoir de la séquence 2 ---------------------------------------
        axes = Axes(
            x_range=[np.log10(200), np.log10(150_000), 1],
            y_range=[-Y_MAX, Y_MAX, 5],
            x_length=10.6,
            y_length=4.4,
            axis_config={"color": P.axe, "stroke_width": 2, "include_ticks": False, "include_tip": False},
            tips=False,
        ).move_to([0.55, -0.95, 0])
        x0, x1 = axes.x_range[0], axes.x_range[1]
        grille = VGroup(
            *[Line(axes.c2p(x0, y), axes.c2p(x1, y), color=P.grille, stroke_width=1) for y in range(-10, 11, 5) if y != 0]
        )
        zero = Line(axes.c2p(x0, 0), axes.c2p(x1, 0), color=P.axe, stroke_width=1.5)
        lab_x = VGroup(*[libelle(fr(v, 0), taille=14).next_to(axes.c2p(np.log10(v), -Y_MAX), DOWN, buff=0.15) for v in GRADUATIONS_N])
        lab_y = VGroup(*[libelle(signe(y), taille=14).next_to(axes.c2p(x0, y), LEFT, buff=0.15) for y in range(-10, 11, 5)])
        titre_x = libelle("taille de l’échantillon (échelle log)", taille=14, couleur=P.discret).next_to(lab_x, DOWN, buff=0.15)
        titre_y = libelle("écart sondage − résultat (points)", taille=14, couleur=P.discret)
        titre_y.next_to(axes.c2p(x0, Y_MAX), UP, buff=0.15).align_to(lab_y, LEFT)

        def position(taille, ecart):
            return axes.c2p(np.log10(taille), np.clip(ecart, -Y_MAX, Y_MAX))

        facteur = ValueTracker(1)
        bord_n = np.geomspace(200, 150_000, 80)

        def marge_50(taille):
            return 100 * Z95 * np.sqrt(0.25 * facteur.get_value() / taille)

        def entonnoir():
            m = np.minimum(marge_50(bord_n), Y_MAX)
            haut_ = [position(t, v) for t, v in zip(bord_n, m)]
            bas_ = [position(t, -v) for t, v in zip(bord_n[::-1], m[::-1])]
            zone = Polygon(*haut_, *bas_, stroke_width=0, fill_color=P.accent, fill_opacity=OPACITE_AIRE)
            # Contour tracé seulement dans le cadre : il part de la taille où la marge atteint le bord (± Y_MAX),
            # sinon la partie écrêtée ressemble à un plateau.
            n_bord = 0.25 * facteur.get_value() * (100 * Z95 / Y_MAX) ** 2
            tailles = np.concatenate([[max(n_bord, bord_n[0])], bord_n[bord_n > n_bord]])
            contour = VGroup(
                *[
                    VMobject().set_points_smoothly([position(t, s * min(marge_50(t), Y_MAX)) for t in tailles]).set_stroke(P.accent, 2)
                    for s in (1, -1)
                ]
            )
            return VGroup(zone, contour)

        forme = always_redraw(entonnoir)
        points = VGroup(*[Dot(position(t, e), radius=0.03) for t, e in zip(n, residu)])

        def couleurs(groupe):
            hors = np.abs(residu) > marge_50(n)
            for d, h in zip(groupe, hors):
                d.set_fill(P.series[0] if h else P.discret, opacity=1 if h else 0.55)

        couleurs(points)
        pos_compteur = axes.c2p(x1, Y_MAX) + DOWN * 0.1
        compteur = always_redraw(
            lambda: VGroup(
                libelle("écarts dans leur marge", taille=15),
                titre("", f"{fr(100 * dedans(facteur.get_value()))}{NBSP}%", "", taille=56),
                libelle(f"visé : {fr(100 * CIBLE, 0)}{NBSP}%", taille=15, couleur=P.discret),
            )
            .arrange(DOWN, aligned_edge=RIGHT, buff=0.08)
            .move_to(pos_compteur, aligned_edge=UP + RIGHT)
        )

        self.play(FadeIn(grille), FadeIn(zero), FadeIn(lab_x), FadeIn(lab_y), FadeIn(titre_x), FadeIn(titre_y), run_time=0.8)
        self.play(FadeIn(forme), FadeIn(points, lag_ratio=0.001), run_time=1.5)
        self.bring_to_back(forme)
        self.bring_to_back(grille)
        self.play(FadeIn(compteur), run_time=0.6)
        self.wait(2)

        # --- On élargit l'entonnoir : c'est celui d'un sondage plus petit ----------------
        cadre = always_redraw(
            lambda: libelle(
                f"entonnoir d’un sondage {fr(facteur.get_value(), 0)} fois plus petit"
                if facteur.get_value() > 1.05
                else "entonnoir de la taille annoncée",
                taille=15,
                couleur=P.texte,
            ).next_to(titre_y, RIGHT, buff=0.6)
        )
        self.play(FadeIn(cadre), run_time=0.5)
        points.add_updater(couleurs)
        for cible in [*ETAPES, facteur_95]:
            self.play(facteur.animate.set_value(cible), run_time=1.8)
            self.wait(1)
        points.clear_updaters()
        self.wait(1)

        # --- La taille de l'entonnoir, c'est une taille de sondage --------------------------
        equivalent = n_reel / facteur_95
        verdict = VGroup(
            libelle(f"un sondage de {fr(n_reel, 0)} personnes a l’entonnoir d’un tirage de", taille=15, couleur=P.texte),
            titre("", f"{fr(entier(equivalent), 0)} personnes", "", taille=48),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.1)
        fond = Rectangle(width=verdict.width + 0.4, height=verdict.height + 0.3, stroke_width=0, fill_color=P.fond, fill_opacity=0.9)
        verdict.move_to(position(150_000, -6), aligned_edge=RIGHT + UP)
        fond.move_to(verdict)
        self.play(FadeIn(fond), FadeIn(verdict, shift=0.1 * UP), run_time=0.8)
        self.wait(2.5)

        entonnoir_partie = VGroup(grille, zero, lab_x, lab_y, titre_x, titre_y, forme, points, compteur, cadre, fond, verdict, source)
        self.play(FadeOut(entonnoir_partie), run_time=0.8)
        for m in (forme, compteur, cadre):
            m.clear_updaters()

        # --- La mesure de l'étude, sondage par sondage ---------------------------------------
        etude = libelle(
            f"la mesure de l’étude, sondage par sondage · {fr(eq['nb_proches'], 0)} sondages des 14 derniers jours · médianes",
            taille=14,
            couleur=P.discret,
        ).next_to(tete, DOWN, buff=0.2, aligned_edge=LEFT)
        lignes = VGroup(
            *[
                VGroup(
                    libelle(nom, taille=16, couleur=P.texte),
                    sous_titre(fr(v, 0), taille=40, couleur=c),
                    libelle(detail, taille=14, couleur=P.discret),
                ).arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
                for nom, v, c, detail in (
                    ("taille annoncée", n_reel, P.texte, "sondages réels"),
                    ("taille équivalente", n_kl, P.series[0], f"÷{NBSP}{fr(n_reel / n_kl, 0)}"),
                    ("témoin", n_temoin, P.texte_2, "un vrai tirage aléatoire par sondage : la méthode retrouve la taille"),
                )
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([-0.5, -0.6, 0])
        self.play(FadeIn(etude), run_time=0.6)
        for ligne in lignes:
            self.play(FadeIn(ligne, shift=0.1 * UP), run_time=0.7)
            self.wait(1)
        self.wait(1.5)
        self.play(FadeOut(VGroup(etude, lignes)), run_time=0.8)

        # --- La taille réelle n'y change presque rien -------------------------------------
        lissage = glissante_equivalents(7)
        courbes = lissage["lignes"]
        source_t = libelle(
            f"{fr(lissage['effectif'], 0)} sondages de la dernière semaine avant le vote · médiane et quartiles, "
            "sondages de taille voisine",
            taille=14,
            couleur=P.discret,
        ).next_to(tete, DOWN, buff=0.2, aligned_edge=LEFT)
        gauche_t, droite_t, bas_t, haut_t = -4.4, 4.6, -2.7, 1.2
        lx_min, lx_max = np.log10(courbes.n.min()), np.log10(courbes.n.max())
        ly_min, ly_max = np.log10(30), np.log10(30_000)

        def pt(x, y):
            return np.array([
                gauche_t + (np.log10(x) - lx_min) / (lx_max - lx_min) * (droite_t - gauche_t),
                bas_t + (np.clip(np.log10(y), ly_min, ly_max) - ly_min) / (ly_max - ly_min) * (haut_t - bas_t),
                0,
            ])

        graduations_y = [100, 1_000, 10_000]
        graduations_x = [v for v in (1_000, 2_000, 5_000) if courbes.n.min() <= v <= courbes.n.max()]
        grille_t = VGroup(
            *[Line(pt(courbes.n.min(), v), pt(courbes.n.max(), v), color=P.grille, stroke_width=1) for v in graduations_y],
            *[Line(pt(v, 30), pt(v, 30_000), color=P.grille, stroke_width=1) for v in graduations_x],
        )
        base_t = Line(pt(courbes.n.min(), 30), pt(courbes.n.max(), 30), color=P.axe, stroke_width=1.5)
        lab_y_t = VGroup(*[libelle(fr(v, 0), taille=14).next_to(pt(courbes.n.min(), v), LEFT, buff=0.15) for v in graduations_y])
        titre_y_t = libelle("taille équivalente (échelle log)", taille=14, couleur=P.discret)
        titre_y_t.next_to(pt(courbes.n.min(), 30_000), UP, buff=0.2).align_to(lab_y_t, LEFT)
        lab_x_t = VGroup(*[libelle(fr(v, 0), taille=14).next_to(pt(v, 30), DOWN, buff=0.18) for v in graduations_x])
        titre_x_t = libelle("taille réelle de l’échantillon (échelle log)", taille=14, couleur=P.discret).next_to(lab_x_t, DOWN, buff=0.15)
        diagonale_t = DashedLine(
            pt(courbes.n.min(), courbes.n.min()), pt(courbes.n.max(), courbes.n.max()),
            color=P.discret, dash_length=0.08, stroke_width=1.5,
        )
        lab_diag_t = libelle("pointillés : taille équivalente = taille réelle", taille=13, couleur=P.discret)
        lab_diag_t.next_to(pt(courbes.n.max(), 30), UP + LEFT, buff=0.12)

        def courbe_et_bande(mesure, couleur):
            haut_ = [pt(x, y) for x, y in zip(courbes.n, courbes[f"{mesure}_q3"])]
            bas_ = [pt(x, y) for x, y in zip(courbes.n[::-1], courbes[f"{mesure}_q1"][::-1])]
            bande = Polygon(*haut_, *bas_, stroke_width=0, fill_color=couleur, fill_opacity=OPACITE_AIRE)
            ligne = VMobject().set_points_smoothly([pt(x, y) for x, y in zip(courbes.n, courbes[f"{mesure}_med"])])
            return bande, ligne.set_stroke(couleur, 4)

        bande_reel, ligne_reel = courbe_et_bande("optimal_kl", P.series[0])
        bande_temoin, ligne_temoin = courbe_et_bande("oneshot", P.texte_2)
        premier, dernier = courbes.iloc[0], courbes.iloc[-1]
        val_reel = VGroup(
            sous_titre(fr(entier(premier.optimal_kl_med), 0), taille=24, couleur=P.series[0]).next_to(ligne_reel.get_start(), UP + RIGHT, buff=0.08),
            sous_titre(fr(entier(dernier.optimal_kl_med), 0), taille=24, couleur=P.series[0]).next_to(ligne_reel.get_end(), RIGHT, buff=0.12),
        )
        val_temoin = VGroup(
            sous_titre(fr(entier(premier.oneshot_med), 0), taille=24, couleur=P.texte_2).next_to(ligne_temoin.get_start(), DOWN + RIGHT, buff=0.08),
            sous_titre(fr(entier(dernier.oneshot_med), 0), taille=24, couleur=P.texte_2).next_to(ligne_temoin.get_end(), RIGHT, buff=0.12),
        )
        nom_reel = libelle("sondages réels", taille=15, couleur=P.series[0]).next_to(pt(1_100, courbes.optimal_kl_q1.min()), DOWN, buff=0.2)
        nom_temoin = libelle("témoin · vrai tirage aléatoire de même taille", taille=15, couleur=P.texte_2)
        nom_temoin.next_to(pt(1_300, courbes.oneshot_q3.max()), UP, buff=0.05)

        self.play(FadeIn(source_t), FadeIn(grille_t), FadeIn(base_t), FadeIn(lab_y_t), FadeIn(titre_y_t), FadeIn(lab_x_t), FadeIn(titre_x_t), run_time=1)
        self.play(Create(diagonale_t), FadeIn(lab_diag_t), run_time=0.8)
        self.play(FadeIn(bande_reel), Create(ligne_reel), FadeIn(nom_reel), run_time=1.5)
        self.play(FadeIn(val_reel), run_time=0.5)
        self.wait(2)
        self.play(FadeIn(bande_temoin), Create(ligne_temoin), FadeIn(nom_temoin), run_time=1.5)
        self.play(FadeIn(val_temoin), run_time=0.5)
        self.wait(2.5)
        self.play(
            FadeOut(VGroup(
                source_t, grille_t, base_t, lab_y_t, titre_y_t, lab_x_t, titre_x_t, diagonale_t, lab_diag_t,
                bande_reel, ligne_reel, bande_temoin, ligne_temoin, val_reel, val_temoin, nom_reel, nom_temoin,
            )),
            run_time=0.8,
        )

        # --- Plus on s'éloigne de l'élection, plus elle baisse --------------------------------
        source_j = libelle(
            f"{fr(eq['effectifs_fenetres']['30'], 0)} sondages du dernier mois avant le vote · taille équivalente médiane et quartiles",
            taille=14,
            couleur=P.discret,
        ).next_to(tete, DOWN, buff=0.2, aligned_edge=LEFT)
        y0, y_haut, v_max = -2.7, 1.2, 700
        xs = np.linspace(-3.5, 4.1, len(jours))

        def y_ecran(v):
            return y0 + v / v_max * (y_haut - y0)

        grille = VGroup(
            *[
                Line([-4.4, y_ecran(v), 0], [5.2, y_ecran(v), 0], color=P.axe if v == 0 else P.grille, stroke_width=1.5 if v == 0 else 1)
                for v in range(0, v_max + 1, 200)
            ]
        )
        lab_y = VGroup(*[libelle(fr(v, 0), taille=14).next_to([-4.4, y_ecran(v), 0], LEFT, buff=0.15) for v in range(0, v_max + 1, 200)])
        titre_y = libelle("taille équivalente", taille=14, couleur=P.discret).next_to([-4.4, y_ecran(v_max), 0], UP, buff=0.2)
        titre_y.align_to(lab_y, LEFT)
        lab_x = VGroup(*[libelle(b["tranche"], taille=15).next_to([x, y0, 0], DOWN, buff=0.18) for b, x in zip(jours, xs)])
        titre_x = libelle("jours avant l’élection", taille=14, couleur=P.discret).next_to(lab_x, DOWN, buff=0.15)

        boites = VGroup()
        for b, x in zip(jours, xs):
            boite = Rectangle(
                width=0.75, height=y_ecran(b["q3"]) - y_ecran(b["q1"]),
                stroke_color=P.series[0], stroke_width=2, fill_color=P.series[0], fill_opacity=OPACITE_AIRE,
            ).move_to([x, (y_ecran(b["q1"]) + y_ecran(b["q3"])) / 2, 0])
            mediane = Line([x - 0.375, y_ecran(b["med"]), 0], [x + 0.375, y_ecran(b["med"]), 0], color=P.series[0], stroke_width=5)
            valeur = sous_titre(fr(entier(b["med"]), 0), taille=30).next_to(mediane, RIGHT, buff=0.12)
            boites.add(VGroup(boite, mediane, valeur))

        self.play(FadeIn(source_j), FadeIn(grille), FadeIn(lab_y), FadeIn(titre_y), FadeIn(lab_x), FadeIn(titre_x), run_time=1)
        for bx in boites:
            self.play(FadeIn(bx, shift=0.1 * UP), run_time=0.45)
        self.wait(2.5)

        # --- Le constat ---------------------------------------------------------------------
        self.play(FadeOut(VGroup(source_j, grille, lab_y, titre_y, lab_x, titre_x, boites)), run_time=0.8)
        constat = titre(f"{fr(n_reel, 0)} sondés, la précision de ", f"{fr(n_kl, 0)}", "", taille=64)
        detail = sous_titre(
            f"un sondage se comporte comme un tirage aléatoire de {fr(n_kl, 0)} personnes · {fr(n_reel / n_kl, 0)} fois moins",
            taille=28,
        )
        temps = sous_titre(
            f"{fr(entier(jours[0]['med']), 0)} dans les {jours[0]['tranche'].split('–')[1]} derniers jours, "
            f"{fr(entier(jours[-1]['med']), 0)} un mois avant",
            taille=28,
            couleur=P.texte_2,
        )
        VGroup(constat, detail, temps).arrange(DOWN, buff=0.35).move_to([0, -0.4, 0])
        self.play(FadeIn(constat, shift=0.15 * UP), run_time=1)
        self.play(FadeIn(detail), run_time=0.6)
        self.wait(0.8)
        self.play(FadeIn(temps), run_time=0.6)
        self.wait(2.5)
