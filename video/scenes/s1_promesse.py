"""Séquence 1 — La promesse : ce que veut dire « ± 3 points ».

Rendu : .venv/Scripts/manim -ql scenes/s1_promesse.py Promesse
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
    AnimationGroup,
    Create,
    DashedLine,
    Dot,
    DoubleArrow,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    MathTex,
    NumberLine,
    Rectangle,
    Transform,
    VGroup,
    Write,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theme import MONO, OPACITE_AIRE, SERIF, P, SceneRR, ecrire, entete, fr, libelle, sous_titre, titre  # noqa: E402

GRAINE = 2016
NB_TIRAGES = 600
NB_RAPIDES = 40  # tirages montrés un par un (les premiers lentement, puis de plus en plus vite)
LOTS = [80, 150, 300, NB_TIRAGES]  # tirages cumulés affichés ensuite, par lots
UNITE_MAX = 0.3  # hauteur d'un tirage dans l'histogramme tant qu'il y a peu de tirages
BORNES = np.arange(44, 56.001, 0.5)  # tranches de l'histogramme, en points de %
Z95 = 1.96
HAUTEUR = 3.0  # hauteur maximale d'une barre
NB_ECHANTILLON = 110  # points mis en évidence dans la population (symbolique)
NB_LENTS = 3  # premiers tirages montrés lentement

NBSP = " "


def marge(n, p=0.5):
    """Marge d'erreur à 95 %, en points de pourcentage."""
    return 100 * Z95 * np.sqrt(p * (1 - p) / n)


def tex_nombre(x, decimales=1):
    """Nombre au format français pour LaTeX (virgule sans espace parasite)."""
    return fr(x, decimales).replace(",", "{,}").replace(" ", r"\,")


def couleur_cote(v):
    """Couleur d'un tirage : celle du vote majoritaire dans l'échantillon (a au-dessus de 50 %, b en dessous)."""
    if v > 50:
        return P.series[0]
    if v < 50:
        return P.series[1]
    return P.texte_2


def puce(couleur, contenu):
    return VGroup(Dot(radius=0.06, color=couleur), libelle(contenu, taille=16)).arrange(RIGHT, buff=0.12)


class Promesse(SceneRR):
    def construct(self):
        rng = np.random.default_rng(GRAINE)
        t1000 = 100 * rng.binomial(1000, 0.5, NB_TIRAGES) / 1000
        t4000 = 100 * rng.binomial(4000, 0.5, NB_TIRAGES) / 4000

        # --- En-tête -------------------------------------------------------
        tete = entete(1, "la promesse", "Ce que promet la ", "marge", f"{NBSP}d’erreur")
        tete.to_corner(UL, buff=0.55)
        self.play(FadeIn(tete, shift=0.15 * DOWN), run_time=1)

        # --- Population ----------------------------------------------------
        cols, rows = 36, 22
        nb = cols * rows
        votes = rng.permutation(np.r_[np.zeros(nb // 2, int), np.ones(nb - nb // 2, int)])
        population = VGroup(*[Dot(radius=0.042, color=P.series[v]) for v in votes])
        population.arrange_in_grid(rows=rows, cols=cols, buff=0.06)
        population.move_to([-3.75, -0.35, 0])
        titre_pop = libelle("population · des millions d’électeurs").next_to(population, UP, buff=0.25, aligned_edge=LEFT)
        legende = VGroup(puce(P.series[0], f"vote a · 50{NBSP}%"), puce(P.series[1], "vote b")).arrange(RIGHT, buff=0.5)
        legende.next_to(population, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(
            FadeIn(titre_pop),
            LaggedStart(*[FadeIn(d, scale=0.3) for d in population], lag_ratio=0.0015, run_time=2),
        )
        self.play(FadeIn(legende), run_time=0.6)

        # --- Axe de l'histogramme ------------------------------------------
        axe = NumberLine(
            x_range=[44, 56, 1], length=7, color=P.axe, stroke_width=2, tick_size=0.05, include_numbers=False
        ).move_to([2.95, -2.55, 0])
        graduations = VGroup(
            *[libelle(f"{v}{NBSP}%", taille=15).next_to(axe.n2p(v), DOWN, buff=0.18) for v in range(44, 57, 2)]
        )
        verite = DashedLine(axe.n2p(50), axe.n2p(50) + UP * (HAUTEUR + 0.55), color=P.texte_2, dash_length=0.08, stroke_width=1.5)
        lab_verite = libelle(f"vraie valeur · 50{NBSP}%", taille=15).next_to(verite, UP, buff=0.1)
        titre_axe = libelle("résultat du tirage (part du vote a)", taille=15, couleur=P.discret)
        titre_axe.next_to(graduations, DOWN, buff=0.2)
        self.play(Create(axe), FadeIn(graduations), FadeIn(titre_axe), run_time=1)
        self.play(Create(verite), FadeIn(lab_verite), run_time=0.8)

        # --- Premiers tirages, un par un -----------------------------------
        taille_ech = libelle("échantillon · 1 000 personnes", couleur=P.texte).next_to(legende, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(taille_ech), run_time=0.5)

        def trait_tirage(v):
            return Line(axe.n2p(v), axe.n2p(v) + UP * 0.35, color=couleur_cote(v), stroke_width=4)

        def envol(choisis, v):
            """Copies des personnes tirées qui rejoignent le point du résultat sur l'axe."""
            cible = axe.n2p(v) + UP * 0.35
            copies = [d.copy() for d in choisis]
            return copies, [c.animate.move_to(cible).scale(0.4).set_opacity(0.6) for c in copies]

        tapis = VGroup()
        for v in t1000[:NB_LENTS]:
            idx = set(rng.choice(nb, NB_ECHANTILLON, replace=False).tolist())
            choisis = [population[j] for j in sorted(idx)]
            reste = VGroup(*[population[j] for j in range(nb) if j not in idx])
            resultat = VGroup(
                libelle("vote a dans l’échantillon", taille=15),
                ecrire(f"{fr(v)}{NBSP}%", SERIF, 56),
            ).arrange(DOWN, buff=0.1)
            resultat.next_to(axe.n2p(50) + UP * 2.3, LEFT, buff=0.3)
            self.play(
                reste.animate.set_opacity(0.15),
                AnimationGroup(*[d.animate.scale(1.8) for d in choisis]),
                FadeIn(resultat, shift=0.1 * UP),
                run_time=1.4,
            )
            self.wait(1.0)
            copies, vols = envol(choisis, v)
            self.play(
                FadeOut(resultat, shift=0.5 * DOWN),
                LaggedStart(*vols, lag_ratio=0.004),
                reste.animate.set_opacity(1),
                AnimationGroup(*[d.animate.scale(1 / 1.8) for d in choisis]),
                run_time=1.4,
            )
            trait = trait_tirage(v)
            self.remove(*copies)
            self.play(Create(trait), run_time=0.4)
            tapis.add(trait)
            self.wait(0.8)

        # --- Les tirages s'accélèrent ----------------------------------------
        def texte_compteur(k):
            return libelle(f"tirages · {k}", couleur=P.texte).next_to(axe, UP, buff=HAUTEUR + 0.9).align_to(axe, RIGHT)

        compteur = texte_compteur(NB_LENTS)
        self.play(FadeIn(compteur), run_time=0.4)
        durees = np.geomspace(0.7, 0.12, NB_RAPIDES - NB_LENTS)
        for k, (v, duree) in enumerate(zip(t1000[NB_LENTS:NB_RAPIDES], durees), start=NB_LENTS + 1):
            idx = rng.choice(nb, NB_ECHANTILLON, replace=False)
            copies, vols = envol([population[j] for j in idx], v)
            self.play(LaggedStart(*vols, lag_ratio=0.003), run_time=duree)
            self.remove(*copies)
            trait = trait_tirage(v)
            nouveau = texte_compteur(k)
            self.add(trait, nouveau)
            self.remove(compteur)
            compteur = nouveau
            tapis.add(trait)
        self.wait(0.6)

        # --- Les traits fusionnent en barres ----------------------------------
        def unite(valeurs):
            """Hauteur d'un tirage : fixe au début, puis réduite pour que la plus haute barre tienne."""
            return min(UNITE_MAX, HAUTEUR / np.histogram(valeurs, BORNES)[0].max())

        def couleur_tranche(k):
            return couleur_cote((BORNES[k] + BORNES[k + 1]) / 2)

        def largeur_tranche(k):
            return axe.n2p(BORNES[k + 1])[0] - axe.n2p(BORNES[k])[0] - 0.04

        def barres(valeurs, ech, marge_pts=None):
            comptes = np.histogram(valeurs, BORNES)[0]
            groupe = VGroup()
            for k, c in enumerate(comptes):
                centre = (BORNES[k] + BORNES[k + 1]) / 2
                estompee = marge_pts is not None and abs(centre - 50) >= marge_pts
                r = Rectangle(
                    width=largeur_tranche(k),
                    height=max(c * ech, 0.002),
                    stroke_width=0,
                    fill_color=couleur_tranche(k),
                    fill_opacity=0.3 if estompee else 1,
                )
                r.move_to(axe.n2p(centre), aligned_edge=DOWN)
                groupe.add(r)
            return groupe

        # Chaque trait se fond dans la barre de sa tranche : les traits d'une même tranche
        # convergent vers le même rectangle et fusionnent.
        histo = barres(t1000[:NB_RAPIDES], unite(t1000[:NB_RAPIDES]))
        tranches = np.digitize(t1000[:NB_RAPIDES], BORNES) - 1
        self.play(
            *[Transform(t, histo[k].copy()) for t, k in zip(tapis, tranches)],
            run_time=1.5,
        )
        self.remove(*tapis)
        self.add(histo)

        # --- Des centaines de tirages -----------------------------------------
        for lot in LOTS:
            self.play(
                Transform(histo, barres(t1000[:lot], unite(t1000[:lot]))),
                Transform(compteur, texte_compteur(lot)),
                run_time=0.9,
            )
        echelle = unite(t1000)
        self.wait(1)

        # --- La marge à 95 % ------------------------------------------------
        m1000 = marge(1000)

        def bande(m, ech_hauteur=HAUTEUR + 0.25):
            gauche, droite = axe.n2p(50 - m), axe.n2p(50 + m)
            zone = Rectangle(width=droite[0] - gauche[0], height=ech_hauteur, stroke_width=0, fill_color=P.accent, fill_opacity=OPACITE_AIRE)
            zone.move_to((gauche + droite) / 2, aligned_edge=DOWN)
            fleche = DoubleArrow(
                gauche + UP * (ech_hauteur + 0.12),
                droite + UP * (ech_hauteur + 0.12),
                buff=0,
                color=P.texte,
                stroke_width=2,
                tip_length=0.14,
                max_tip_length_to_length_ratio=0.2,
            )
            valeur = ecrire(f"±{NBSP}{fr(m)}{NBSP}points", SERIF, 34).next_to(fleche, UP, buff=0.1)
            return VGroup(zone, fleche, valeur)

        marge_1000 = bande(m1000)
        part = sous_titre(f"95{NBSP}% des tirages", taille=28, couleur=P.texte_2).next_to(marge_1000[2], UP, buff=0.08)
        self.play(FadeOut(lab_verite), FadeOut(compteur), FadeIn(marge_1000[0]), run_time=0.6)
        self.bring_to_back(marge_1000[0])
        self.play(
            Transform(histo, barres(t1000, echelle, m1000)),
            Create(marge_1000[1]),
            FadeIn(marge_1000[2]),
            FadeIn(part),
            run_time=1.2,
        )
        self.wait(2)

        # --- La formule -----------------------------------------------------
        formule = MathTex(r"\sigma = \sqrt{\frac{p\,(1-p)}{n}}", color=P.texte, font_size=46)
        marge_tex = MathTex(r"\text{marge à 95\,\%} = 1{,}96\,\sigma", color=P.texte, font_size=40)
        calcul_1000 = MathTex(
            r"n = 1\,000 \;\Rightarrow\; \pm\," + tex_nombre(m1000) + r"\ \text{points}", color=P.texte, font_size=40
        )
        bloc = VGroup(formule, marge_tex, calcul_1000).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        bloc.move_to([-3.75, 0.25, 0])
        self.play(FadeOut(VGroup(population, titre_pop, legende, taille_ech)), run_time=0.6)
        self.play(Write(formule), run_time=1.2)
        self.play(FadeIn(marge_tex, shift=0.1 * DOWN), run_time=0.8)
        self.play(FadeIn(calcul_1000, shift=0.1 * DOWN), run_time=0.8)
        self.wait(1.5)

        # --- Quatre fois plus de monde --------------------------------------
        m4000 = marge(4000)
        echelle_4000 = HAUTEUR / np.histogram(t4000, BORNES)[0].max()
        calcul_4000 = MathTex(
            r"n = 4\,000 \;\Rightarrow\; \pm\," + tex_nombre(m4000) + r"\ \text{points}", color=P.accent, font_size=40
        ).next_to(calcul_1000, DOWN, aligned_edge=LEFT, buff=0.35)
        marge_4000 = bande(m4000)
        regle = sous_titre(f"4 × plus de monde{NBSP}: marge ÷{NBSP}2", taille=28).next_to(calcul_4000, DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(
            Transform(histo, barres(t4000, echelle_4000, m4000)),
            Transform(marge_1000, marge_4000),
            part.animate.next_to(marge_4000[2], UP, buff=0.08),
            FadeIn(calcul_4000, shift=0.1 * DOWN),
            run_time=1.6,
        )
        self.play(FadeIn(regle, shift=0.1 * UP), run_time=0.8)
        self.wait(2)

        # --- La promesse, en une phrase -------------------------------------
        tout = VGroup(axe, graduations, titre_axe, verite, histo, marge_1000, part, bloc, calcul_4000, regle)
        self.play(FadeOut(tout), run_time=0.8)
        promesse = titre(f"±{NBSP}3 points, ", "19 fois", " sur 20", taille=72)
        precision = sous_titre(f"pour 1 000 personnes interrogées et un candidat à 50{NBSP}%", taille=34, couleur=P.texte_2)
        exception = sous_titre(f"1 fois sur 20{NBSP}: plus de 3 points d’écart", taille=34, couleur=P.texte_2)
        VGroup(promesse, precision, exception).arrange(DOWN, buff=0.35).move_to([0, 0.2, 0])
        self.play(FadeIn(promesse, shift=0.15 * UP), run_time=1)
        self.play(FadeIn(precision), run_time=0.6)
        self.play(FadeIn(exception), run_time=0.6)
        self.wait(1.2)

        invite = ecrire(
            f"<span foreground='{P.accent.to_hex()}'>&gt; </span>vérifions-la<span foreground='{P.accent.to_hex()}'> █</span>",
            MONO,
            30,
            markup=True,
        ).next_to(exception, DOWN, buff=0.8)
        self.play(FadeIn(invite), run_time=0.5)
        curseur = invite[-1]
        for _ in range(3):
            self.play(curseur.animate.set_opacity(0), run_time=0.25)
            self.play(curseur.animate.set_opacity(1), run_time=0.25)
        self.wait(0.5)
