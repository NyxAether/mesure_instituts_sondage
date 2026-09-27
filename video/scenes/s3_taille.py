"""Séquence 3 — L'excédent d'erreur au-dessus de la théorie ne diminue pas avec la taille de l'échantillon.

Rendu : .venv/Scripts/python.exe -m manim -ql scenes/s3_taille.py Taille
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
    Axes,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    Polygon,
    Rectangle,
    Transform,
    VGroup,
    VMobject,
    ValueTracker,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donnees import TOUS  # noqa: E402
from theme import P, SceneRR, entete, fr, libelle, sous_titre, titre  # noqa: E402

NBSP = " "
Y_ECART = 14  # points ; les écarts exportés sont bornés à ± 14
Y_MOYENNE = 2.6  # haut de l'axe une fois zoomé sur les moyennes
GRADUATIONS_N = [300, 1_000, 3_000, 10_000, 30_000, 100_000]
X_RANGE = [np.log10(200), np.log10(150_000), 1]


def signe(v):
    return f"+{fr(v, 0)}" if v > 0 else ("0" if v == 0 else f"−{fr(-v, 0)}")


class Taille(SceneRR):
    def construct(self):
        nuage = TOUS["nuage"]
        pts = nuage["points"]
        n = np.array([p["n"] for p in pts])
        residu = 100 * np.array([p["residu"] for p in pts])
        tranches = TOUS["par_taille"]
        n_med = np.array([t["n"] for t in tranches])
        obs = 100 * np.array([t["obs"] for t in tranches])
        th = 100 * np.array([t["th"] for t in tranches])
        membres = [np.flatnonzero((n >= t["n_min"]) & (n <= t["n_max"])) for t in tranches]

        assert sum(len(m) for m in membres) == len(pts), "tranches incomplètes"
        for m, o in zip(membres, obs):
            assert np.isclose(np.abs(residu[m]).mean(), o, atol=0.005), "moyenne de tranche différente de l'export"

        # --- En-tête ----------------------------------------------------------
        tete = entete(3, "la taille", "L’erreur selon la ", "taille", f"{NBSP}de l’échantillon").to_corner(UL, buff=0.55)
        source = libelle(
            f"les mêmes {fr(nuage['nb_lignes'], 0)} intentions de vote · regroupées en {len(tranches)} tranches de taille",
            taille=14,
            couleur=P.discret,
        ).next_to(tete, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(tete, shift=0.15 * DOWN), run_time=1)
        self.play(FadeIn(source), run_time=0.6)

        # --- Trois repères de même cadre : écart signé, erreur, erreur moyenne ------
        def repere(y_min, y_max):
            return Axes(
                x_range=X_RANGE,
                y_range=[y_min, y_max, 1],
                x_length=10.6,
                y_length=4.4,
                axis_config={"include_ticks": False, "include_tip": False},
                tips=False,
            ).move_to([0.55, -0.95, 0])

        a_ecart, a_erreur, a_moyenne = repere(-Y_ECART, Y_ECART), repere(0, Y_ECART), repere(0, Y_MOYENNE)

        def pos(axes, taille, y):
            return axes.c2p(np.log10(taille), y)

        # Axe vertical animable : ses bornes sont des ValueTracker, et la grille glisse avec l'échelle
        # (les lignes fines apparaissent quand elles s'espacent, celles qui sortent du cadre s'effacent).
        bas, haut, visible = ValueTracker(-Y_ECART), ValueTracker(Y_ECART), ValueTracker(0)
        x_gauche, y_base, _ = a_erreur.c2p(X_RANGE[0], 0)
        x_droite = a_erreur.c2p(X_RANGE[1], 0)[0]
        hauteur = a_erreur.c2p(X_RANGE[0], Y_ECART)[1] - y_base

        def y_ecran(y):
            b, h = bas.get_value(), haut.get_value()
            return y_base + (y - b) / (h - b) * hauteur

        def pas(y):
            return 5 if y % 5 == 0 else (1 if y % 1 == 0 else 0.5)

        valeurs = np.arange(-Y_ECART, Y_ECART + 0.01, 0.5)
        lignes = VGroup(
            *[Line(LEFT, RIGHT, color=P.axe if y == 0 else P.grille, stroke_width=1.5 if y == 0 else 1) for y in valeurs]
        )
        entieres = [y for y in valeurs if y % 1 == 0]
        signees = VGroup(*[libelle(signe(y), taille=14) for y in entieres])
        simples = VGroup(*[libelle(fr(y, 0), taille=14) for y in entieres])

        def grille_a_jour(_):
            b, h, v = bas.get_value(), haut.get_value(), visible.get_value()
            etendue = h - b
            signe_visible = np.clip(-b / 3, 0, 1)  # graduations « +5 / −5 » tant que l'axe descend sous zéro
            opacites = {}
            for y, ligne in zip(valeurs, lignes):
                ye = y_ecran(y)
                bord = np.clip(1 - max(y - h, b - y) / (0.05 * etendue), 0, 1)
                espace = pas(y) * hauteur / etendue
                o = v * bord * (1 if y == 0 else np.clip((espace - 0.3) / 0.3, 0, 1))
                ligne.put_start_and_end_on([x_gauche, ye, 0], [x_droite, ye, 0]).set_stroke(opacity=o)
                opacites[y] = v * bord * np.clip((espace - 0.5) / 0.3, 0, 1)
            for y, ls, lu in zip(entieres, signees, simples):
                for lab, poids in ((ls, signe_visible), (lu, 1 - signe_visible)):
                    lab.next_to([x_gauche, y_ecran(y), 0], LEFT, buff=0.15).set_opacity(opacites[y] * poids)

        grille = VGroup(lignes, signees, simples)
        grille.add_updater(grille_a_jour)
        grille_a_jour(grille)

        def titre_axe(contenu):
            t = libelle(contenu, taille=14, couleur=P.discret)
            return t.next_to([x_gauche, y_base + hauteur, 0], UP, buff=0.15).align_to([x_gauche - 0.45, 0, 0], LEFT)

        titre_ecart = titre_axe("écart sondage − résultat (points)")
        titre_erreur = titre_axe("erreur, sans son signe (points)")
        titre_moyenne = titre_axe("erreur moyenne de la tranche (points)")

        lab_x = VGroup(
            *[libelle(fr(v, 0), taille=14).next_to(pos(a_ecart, v, -Y_ECART), DOWN, buff=0.15) for v in GRADUATIONS_N]
        )
        titre_x = libelle("taille de l’échantillon (échelle log)", taille=14, couleur=P.discret).next_to(lab_x, DOWN, buff=0.15)

        # --- Les écarts de la séquence 2 ------------------------------------------
        def nuage_points(axes, ecarts):
            return VGroup(
                *[
                    Dot(pos(axes, t, np.clip(e, axes.y_range[0], axes.y_range[1])), radius=0.03, color=P.discret, fill_opacity=0.55)
                    for t, e in zip(n, ecarts)
                ]
            )

        points = nuage_points(a_ecart, residu)
        self.add(grille)
        self.play(visible.animate.set_value(1), FadeIn(titre_ecart), FadeIn(lab_x), FadeIn(titre_x), run_time=0.8)
        self.play(FadeIn(points, lag_ratio=0.001), run_time=1.5)
        self.wait(1)

        # --- On oublie le signe : les écarts négatifs se replient vers le haut ------
        self.play(
            Transform(points, nuage_points(a_erreur, np.abs(residu))),
            bas.animate.set_value(0),
            FadeOut(titre_ecart),
            FadeIn(titre_erreur),
            run_time=1.8,
        )
        self.wait(0.8)

        # --- Tranches de taille ----------------------------------------------------
        bandes = VGroup()
        for k, t in enumerate(tranches):
            gauche, droite = pos(a_erreur, max(t["n_min"], 200), 0), pos(a_erreur, t["n_max"], Y_ECART)
            bande = Rectangle(
                width=max(droite[0] - gauche[0], 0.04),
                height=droite[1] - gauche[1],
                stroke_width=0,
                fill_color=P.texte,
                fill_opacity=0.06 if k % 2 == 0 else 0,
            )
            bande.move_to((gauche + droite) / 2)
            bandes.add(bande)
        self.play(FadeIn(bandes), run_time=0.8)
        self.bring_to_back(bandes)
        self.wait(0.6)

        # --- Chaque tranche se réduit à son erreur moyenne ---------------------------
        par_tranche = [VGroup(*[points[j] for j in m]) for m in membres]
        moyennes = VGroup(*[Dot(pos(a_erreur, t, o), radius=0.075, color=P.series[0]) for t, o in zip(n_med, obs)])
        self.play(
            *[
                Transform(groupe, VGroup(*[Dot(pos(a_erreur, t, o), radius=0.075, color=P.series[0]) for _ in groupe]))
                for groupe, t, o in zip(par_tranche, n_med, obs)
            ],
            run_time=2,
        )
        self.add(moyennes)
        self.remove(points, *par_tranche)
        self.wait(0.6)

        # --- Zoom sur les moyennes ---------------------------------------------------
        for d, t, o in zip(moyennes, n_med, obs):
            d.add_updater(lambda m, t=t, o=o: m.move_to([pos(a_erreur, t, 0)[0], y_ecran(o), 0]))
        self.play(FadeOut(bandes), FadeOut(titre_erreur), run_time=0.5)
        self.play(haut.animate.set_value(Y_MOYENNE), FadeIn(titre_moyenne), run_time=2.5)
        for d in moyennes:
            d.clear_updaters()
        self.wait(0.5)

        # --- En théorie : l'erreur devrait fondre ------------------------------------
        sommets_th = [pos(a_moyenne, t, e) for t, e in zip(n_med, th)]
        courbe_th = VGroup(
            *[DashedLine(a, b, color=P.texte_2, stroke_width=3, dash_length=0.08) for a, b in zip(sommets_th, sommets_th[1:])]
        )
        points_th = VGroup(*[Dot(s, radius=0.06, color=P.texte_2) for s in sommets_th])
        lab_th = libelle("attendu en théorie", taille=15, couleur=P.texte_2).next_to(sommets_th[-1], DOWN, buff=0.25)
        valeurs_th = VGroup(
            libelle(fr(th[0]), taille=15, couleur=P.texte_2).next_to(sommets_th[0], DOWN + LEFT, buff=0.1),
            libelle(fr(th[-1]), taille=15, couleur=P.texte_2).next_to(sommets_th[-1], RIGHT, buff=0.15),
        )
        self.play(Create(courbe_th), FadeIn(points_th), run_time=1.8)
        self.play(FadeIn(lab_th), FadeIn(valeurs_th), run_time=0.8)
        self.wait(1.5)

        # --- En réalité : elle reste bloquée ------------------------------------------
        sommets_obs = [pos(a_moyenne, t, o) for t, o in zip(n_med, obs)]
        courbe_obs = VMobject().set_points_as_corners(sommets_obs).set_stroke(P.series[0], 4)
        lab_obs = libelle("observé", taille=15, couleur=P.series[0]).next_to(sommets_obs[-1], UP, buff=0.25)
        valeurs_obs = VGroup(
            libelle(fr(obs[0]), taille=15, couleur=P.series[0]).next_to(sommets_obs[0], UP + LEFT, buff=0.1),
            libelle(fr(obs[-1]), taille=15, couleur=P.series[0]).next_to(sommets_obs[-1], RIGHT, buff=0.15),
        )
        self.play(Create(courbe_obs), run_time=1.5)
        self.bring_to_front(moyennes)
        self.play(FadeIn(lab_obs), FadeIn(valeurs_obs), run_time=0.8)
        self.wait(1.5)

        # --- L'écart entre les deux ----------------------------------------------------
        ecart = Polygon(*sommets_obs, *reversed(sommets_th), stroke_width=0, fill_color=P.series[0], fill_opacity=0.12)
        self.play(FadeIn(ecart), run_time=1)
        self.bring_to_back(ecart)
        self.bring_to_back(grille)

        def rapport(k, cote):
            milieu = (sommets_obs[k] + sommets_th[k]) / 2
            return titre("", f"×{NBSP}{fr(obs[k] / th[k])}", "", taille=40).next_to(milieu, cote, buff=0.35)

        rapport_gauche, rapport_droit = rapport(0, LEFT), rapport(len(tranches) - 1, RIGHT)
        self.play(FadeIn(rapport_gauche, shift=0.1 * UP), run_time=0.7)
        self.wait(0.8)
        self.play(FadeIn(rapport_droit, shift=0.1 * UP), run_time=0.7)
        self.wait(2)

        # --- Ce qui dépasse la théorie est le même à toutes les tailles ------------------
        exces = obs - th
        ecarts = VGroup(*[Line(b, h, color=P.series[0], stroke_width=5) for b, h in zip(sommets_th, sommets_obs)])
        lab_exces = VGroup(
            libelle(f"{fr(exces.min())} à {fr(exces.max())} point au-dessus de la théorie", taille=15, couleur=P.series[0]),
            libelle("à toutes les tailles", taille=15, couleur=P.series[0]),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.1)
        lab_exces.move_to(pos(a_moyenne, 10 ** X_RANGE[1], 2.45), aligned_edge=RIGHT)
        self.play(
            FadeOut(rapport_gauche),
            FadeOut(rapport_droit),
            ecart.animate.set_fill(opacity=0.05),
            LaggedStart(*[Create(e) for e in ecarts], lag_ratio=0.15),
            run_time=1.5,
        )
        self.play(FadeIn(lab_exces), run_time=0.7)
        self.wait(2.5)

        # --- Le constat -------------------------------------------------------------------
        graphique = VGroup(
            titre_moyenne, lab_x, titre_x, source, ecart, courbe_th, points_th, lab_th, valeurs_th,
            courbe_obs, moyennes, lab_obs, valeurs_obs, ecarts, lab_exces,
        )
        self.play(FadeOut(graphique), visible.animate.set_value(0), run_time=0.8)
        grille.clear_updaters()
        self.remove(grille)
        constat = titre("L’excédent d’erreur ", "ne diminue pas", "", taille=64)
        detail = sous_titre(
            f"{fr(exces.min())} à {fr(exces.max())} point au-dessus de la théorie, à toutes les tailles", taille=30
        )
        conclusion = sous_titre(
            f"seule la baisse prévue par le hasard a lieu{NBSP}: −{fr(obs[0] - obs[-1])} point, "
            f"contre −{fr(th[0] - th[-1])} en théorie",
            taille=28,
            couleur=P.texte_2,
        )
        VGroup(constat, detail, conclusion).arrange(DOWN, buff=0.35).move_to([0, -0.4, 0])
        self.play(FadeIn(constat, shift=0.15 * UP), run_time=1)
        self.play(FadeIn(detail), run_time=0.6)
        self.wait(0.8)
        self.play(FadeIn(conclusion), run_time=0.6)
        self.wait(2.5)
