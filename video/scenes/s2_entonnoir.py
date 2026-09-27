"""Séquence 2 — L'entonnoir : 45 % des écarts hors de leur marge, contre 5 % attendus en théorie.

Rendu : .venv/Scripts/manim -ql scenes/s2_entonnoir.py Entonnoir
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
    Square,
    Transform,
    VMobject,
    ValueTracker,
    VGroup,
    always_redraw,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donnees import TOUS  # noqa: E402
from theme import OPACITE_AIRE, P, SceneRR, entete, fr, libelle, sous_titre, titre  # noqa: E402

GRAINE = 2002
Z95 = 1.96
NBSP = " "
Y_MAX = 14  # points ; les écarts exportés sont bornés à ± 14
GRADUATIONS_N = [300, 1_000, 3_000, 10_000, 30_000, 100_000]
# Point annoté : une ligne du nuage (France, 2002), retrouvée dans les données au lancement.
EXEMPLE = {"pays": "France", "annee": 2002, "n": 1000.0, "vote": 0.1686, "poll": 0.13}


def signe(v):
    return f"+{fr(v, 0)}" if v > 0 else ("0" if v == 0 else f"−{fr(-v, 0)}")


class Entonnoir(SceneRR):
    def construct(self):
        rng = np.random.default_rng(GRAINE)
        nuage = TOUS["nuage"]
        pts = nuage["points"]
        n = np.array([p["n"] for p in pts])
        vote = np.array([p["vote"] for p in pts])
        residu = 100 * np.array([p["residu"] for p in pts])
        marge_propre = 100 * Z95 * np.sqrt(vote * (1 - vote) / n)
        hors = np.abs(residu) > marge_propre
        # Écarts qu'auraient donnés de vrais tirages aléatoires de même taille, sur le même résultat.
        simule = 100 * (rng.binomial(n.astype(int), vote) / n - vote)
        hors_simule = np.abs(simule) > marge_propre
        # Simplification visuelle : les points sont colorés selon l'entonnoir tracé (marge à p = 50 %),
        # alors que les pourcentages affichés comparent chaque écart à sa propre marge (celle de l'export).
        marge_50 = 100 * Z95 * np.sqrt(0.25 / n)

        assert np.isclose(hors.mean(), nuage["part_hors_marge"]), "part hors marge différente de l'export"
        assert any(all(p[k] == v for k, v in EXEMPLE.items()) for p in pts), "point d'exemple absent des données"

        # --- En-tête et source ------------------------------------------------
        tete = entete(2, "l’entonnoir", "Les sondages face aux ", "résultats").to_corner(UL, buff=0.55)
        source = libelle(
            f"jennings & wlezien · {fr(nuage['nb_lignes'], 0)} intentions de vote · "
            f"{fr(nuage['nb_sondages'], 0)} sondages · {nuage['nb_pays']} pays · dernière semaine · depuis {TOUS['annee_min']}",
            taille=14,
            couleur=P.discret,
        ).next_to(tete, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(tete, shift=0.15 * DOWN), run_time=1)
        self.play(FadeIn(source), run_time=0.8)

        # --- Axes -------------------------------------------------------------
        axes = Axes(
            x_range=[np.log10(200), np.log10(150_000), 1],
            y_range=[-Y_MAX, Y_MAX, 5],
            x_length=10.6,
            y_length=4.4,
            axis_config={"color": P.axe, "stroke_width": 2, "include_ticks": False, "include_tip": False},
            tips=False,
        ).move_to([0.55, -0.95, 0])
        grille = VGroup(
            *[
                Line(axes.c2p(axes.x_range[0], y), axes.c2p(axes.x_range[1], y), color=P.grille, stroke_width=1)
                for y in range(-10, 11, 5)
                if y != 0
            ]
        )
        zero = Line(axes.c2p(axes.x_range[0], 0), axes.c2p(axes.x_range[1], 0), color=P.axe, stroke_width=1.5)
        lab_x = VGroup(
            *[
                libelle(fr(v, 0), taille=14).next_to(axes.c2p(np.log10(v), -Y_MAX), DOWN, buff=0.15)
                for v in GRADUATIONS_N
            ]
        )
        lab_y = VGroup(
            *[libelle(signe(y), taille=14).next_to(axes.c2p(axes.x_range[0], y), LEFT, buff=0.15) for y in range(-10, 11, 5)]
        )
        titre_x = libelle("taille de l’échantillon (échelle log)", taille=14, couleur=P.discret).next_to(lab_x, DOWN, buff=0.15)
        titre_y = libelle("écart sondage − résultat (points)", taille=14, couleur=P.discret)
        titre_y.next_to(axes.c2p(axes.x_range[0], Y_MAX), UP, buff=0.15).align_to(lab_y, LEFT)
        self.play(
            FadeIn(grille),
            Create(zero),
            FadeIn(lab_x),
            FadeIn(lab_y),
            FadeIn(titre_x),
            FadeIn(titre_y),
            run_time=1.2,
        )

        def position(taille, ecart):
            return axes.c2p(np.log10(taille), np.clip(ecart, -Y_MAX, Y_MAX))

        # --- Un point, expliqué -------------------------------------------------
        ex = EXEMPLE
        ecart_ex = 100 * (ex["poll"] - ex["vote"])
        marge_ex = 100 * Z95 * np.sqrt(ex["vote"] * (1 - ex["vote"]) / ex["n"])
        point_ex = Dot(position(ex["n"], ecart_ex), radius=0.08, color=P.series[0])
        barre_ex = Line(position(ex["n"], -marge_ex), position(ex["n"], marge_ex), color=P.texte_2, stroke_width=3)
        chapeaux = VGroup(
            *[
                Line(position(ex["n"], s * marge_ex) + LEFT * 0.08, position(ex["n"], s * marge_ex) + RIGHT * 0.08, color=P.texte_2, stroke_width=3)
                for s in (-1, 1)
            ]
        )
        note = VGroup(
            libelle(f"{ex['pays'].lower()} · {ex['annee']} · {fr(ex['n'], 0)} sondés", taille=15, couleur=P.texte),
            libelle(f"sondage {fr(100 * ex['poll'])}{NBSP}% · résultat {fr(100 * ex['vote'])}{NBSP}%", taille=15),
            libelle(f"écart {fr(ecart_ex).replace('-', '−')} pts · marge ±{NBSP}{fr(marge_ex)} pts", taille=15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        note.next_to(point_ex, RIGHT, buff=0.5).shift(DOWN * 0.4)
        lien = DashedLine(point_ex.get_right(), note.get_left(), color=P.discret, stroke_width=1, dash_length=0.05)
        self.play(FadeIn(point_ex, scale=0.5), run_time=0.6)
        self.play(Create(lien), FadeIn(note, shift=0.1 * RIGHT), run_time=0.9)
        self.play(Create(barre_ex), FadeIn(chapeaux), run_time=0.8)
        self.wait(2.5)
        self.play(FadeOut(VGroup(point_ex, barre_ex, chapeaux, note, lien)), run_time=0.6)

        # --- L'entonnoir théorique ---------------------------------------------
        bord = [(m["n"], 100 * m["m"]) for m in nuage["marge"]]
        haut = [position(t, m) for t, m in bord]
        bas = [position(t, -m) for t, m in reversed(bord)]
        zone = Polygon(*haut, *bas, stroke_width=0, fill_color=P.accent, fill_opacity=OPACITE_AIRE)
        contour = VGroup(
            VMobject().set_points_smoothly(haut).set_stroke(P.accent, 2),
            VMobject().set_points_smoothly(bas).set_stroke(P.accent, 2),
        )
        lab_zone = VGroup(
            Square(0.2, stroke_color=P.accent, stroke_width=2, fill_color=P.accent, fill_opacity=OPACITE_AIRE),
            libelle(f"marge d’erreur à 95{NBSP}%", taille=14, couleur=P.texte_2),
        ).arrange(RIGHT, buff=0.15)
        lab_zone.move_to(position(10 ** axes.x_range[1], -9), aligned_edge=RIGHT)
        self.play(FadeIn(zone), Create(contour), FadeIn(lab_zone), run_time=1.2)
        self.bring_to_back(zone)
        self.bring_to_back(grille)

        # --- En théorie : tirages simulés -----------------------
        def couleur(est_hors):
            return P.series[0] if est_hors else P.discret

        def nuage_points(ecarts, est_hors):
            return VGroup(
                *[
                    Dot(position(t, e), radius=0.03, color=couleur(h), fill_opacity=1 if h else 0.55)
                    for t, e, h in zip(n, ecarts, est_hors)
                ]
            )

        points = nuage_points(simule, np.abs(simule) > marge_50)
        reels = nuage_points(residu, np.abs(residu) > marge_50)

        part = ValueTracker(100 * hors_simule.mean())
        pos_compteur = axes.c2p(axes.x_range[1], Y_MAX) + DOWN * 0.1

        def compteur():
            return VGroup(
                libelle("hors de leur marge d’erreur", taille=15),
                titre("", f"{fr(part.get_value())}{NBSP}%", "", taille=56),
            ).arrange(DOWN, aligned_edge=RIGHT, buff=0.08).move_to(pos_compteur, aligned_edge=UP + RIGHT)

        cadre = libelle("en théorie : des tirages aléatoires de même taille", taille=15, couleur=P.texte)
        cadre.next_to(titre_y, RIGHT, buff=0.6)
        affichage = always_redraw(compteur)
        self.play(FadeIn(cadre), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in points], lag_ratio=0.0015, run_time=2.5))
        self.play(FadeIn(affichage), run_time=0.6)
        self.wait(2)

        # --- La réalité ---------------------------------------------------------
        attendu = libelle(f"attendu : 5{NBSP}%", taille=15, couleur=P.discret)
        attendu.next_to(pos_compteur + DOWN * 1.15, DOWN, buff=0).align_to(pos_compteur, RIGHT)
        cadre_reel = libelle("la réalité : les sondages publiés", taille=15, couleur=P.texte).move_to(cadre, aligned_edge=LEFT)
        self.play(Transform(cadre, cadre_reel), FadeIn(attendu), run_time=0.6)
        self.play(
            Transform(points, reels),
            part.animate.set_value(100 * hors.mean()),
            run_time=3,
        )
        self.wait(2.5)

        # --- Le constat -----------------------------------------------------------
        graphique = VGroup(grille, zero, lab_x, lab_y, titre_x, titre_y, zone, contour, lab_zone, points, attendu, cadre, source)
        self.play(FadeOut(graphique), FadeOut(affichage), run_time=0.8)
        constat = titre("", f"{fr(100 * hors.mean(), 0)}{NBSP}%", "", taille=110)
        phrase = sous_titre("des écarts sortent de leur marge d’erreur", taille=40)
        rappel = sous_titre(f"au lieu des 5{NBSP}% attendus en théorie · presque un sur deux", taille=32, couleur=P.texte_2)
        VGroup(constat, phrase, rappel).arrange(DOWN, buff=0.35).move_to([0, -0.4, 0])
        self.play(FadeIn(constat, shift=0.15 * UP), run_time=1)
        self.play(FadeIn(phrase), run_time=0.6)
        self.wait(0.8)
        self.play(FadeIn(rappel), run_time=0.6)
        self.wait(2.5)
