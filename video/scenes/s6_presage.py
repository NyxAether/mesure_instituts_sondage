"""Séquence 6 — Une prédiction plus qu'une photographie, un présage plus qu'une prédiction.

Rendu : .venv/Scripts/python.exe -m manim -ql scenes/s6_presage.py Presage
"""
import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    Cross,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    VGroup,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donnees import FRANCE, TOUS  # noqa: E402
from theme import OPACITE_AIRE, P, SceneRR, entete, fr, libelle, sous_titre, titre  # noqa: E402

NBSP = " "


def citation(texte_, auteur, taille=26):
    """Citation en Newsreader, auteur et source en libellé dessous, filet prune à gauche."""
    corps = VGroup(
        sous_titre(f"« {texte_} »", taille=taille, couleur=P.texte),
        libelle(auteur, taille=13, couleur=P.discret),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    filet = Line(ORIGIN, DOWN * corps.height, color=P.accent, stroke_width=3)
    return VGroup(filet, corps).arrange(RIGHT, buff=0.25, aligned_edge=UP)


class Presage(SceneRR):
    def construct(self):
        source, selection = TOUS["source"], TOUS["selection"]
        tailles = TOUS["par_taille"]

        tete = entete(6, "photographie, prédiction, présage", "Un ", "présage", " plus qu’une prédiction").to_corner(UL, buff=0.55)
        self.play(FadeIn(tete, shift=0.15 * DOWN), run_time=1)

        # --- Précautions ----------------------------------------------------------------
        part_france = FRANCE["selection"]["sondages"] / selection["sondages"]
        largeur = 9.0
        barre = Rectangle(width=largeur, height=0.35, stroke_width=0, fill_color=P.muet, fill_opacity=0.6)
        barre_fr = Rectangle(width=largeur * part_france, height=0.35, stroke_width=0, fill_color=P.series[0], fill_opacity=1)
        barre_fr.align_to(barre, LEFT)
        lab_tous = libelle(
            f"{fr(selection['sondages'], 0)} sondages · {selection['pays']} pays · élections de {TOUS['annee_min']} à {source['fin']}",
            taille=15,
        ).next_to(barre, UP, buff=0.12, aligned_edge=LEFT)
        lab_fr = libelle(
            f"france : {fr(FRANCE['selection']['sondages'], 0)} sondages, {fr(100 * part_france, 0)}{NBSP}%",
            taille=15, couleur=P.series[0],
        ).next_to(barre, DOWN, buff=0.12, aligned_edge=LEFT)
        schema = VGroup(lab_tous, VGroup(barre, barre_fr), lab_fr)
        limites = VGroup(*[
            sous_titre(t, taille=26)
            for t in (
                f"une base qui s’arrête en {source['fin']}",
                "peu de sondages français",
                "des sondages d’un même jour parfois fusionnés en une moyenne",
            )
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        conclusion = sous_titre(
            f"des tendances solides sur {selection['pays']} pays, indicatives pour un pays pris seul",
            taille=26, couleur=P.texte_2,
        )
        VGroup(schema, limites, conclusion).arrange(DOWN, aligned_edge=LEFT, buff=0.55).next_to(tete, DOWN, buff=0.7, aligned_edge=LEFT)
        self.play(FadeIn(lab_tous), FadeIn(barre), run_time=0.8)
        self.play(FadeIn(barre_fr), FadeIn(lab_fr), run_time=0.8)
        for ligne in limites:
            self.play(FadeIn(ligne, shift=0.1 * UP), run_time=0.5)
            self.wait(0.4)
        self.play(FadeIn(conclusion), run_time=0.6)
        self.wait(2)
        self.play(FadeOut(VGroup(schema, limites, conclusion)), run_time=0.8)

        # --- Le hasard, plus petite part de l'erreur ----------------------------------------
        # Pour chaque tranche de taille : erreur typique attendue au hasard (th) et observée (obs), en points.
        haut_max = 2.7
        e_max = max(t["obs"] for t in tailles)
        base_y, x0, pas = -2.9, -5.0, 1.45
        lb = 0.42
        barres, rapports, lab_n = VGroup(), VGroup(), VGroup()
        for i, t in enumerate(tailles):
            x = x0 + i * pas
            h_th = t["th"] / e_max * haut_max
            h_obs = t["obs"] / e_max * haut_max
            th = Rectangle(width=lb, height=h_th, stroke_width=0, fill_color=P.accent, fill_opacity=OPACITE_AIRE * 4)
            th.move_to([x, base_y + h_th / 2, 0])
            obs = Rectangle(width=lb, height=h_obs, stroke_width=0, fill_color=P.texte_2, fill_opacity=0.85)
            obs.move_to([x + lb + 0.04, base_y + h_obs / 2, 0])
            barres.add(VGroup(th, obs))
            rapports.add(sous_titre(f"×{fr(t['obs'] / t['th'])}", taille=22).next_to(obs, UP, buff=0.1).shift(LEFT * (lb + 0.04) / 2))
            lab_n.add(libelle(fr(t["n"], 0), taille=13).next_to([x + (lb + 0.04) / 2, base_y, 0], DOWN, buff=0.12))
        axe = Line([x0 - 0.5, base_y, 0], [x0 + (len(tailles) - 1) * pas + lb + 0.5, base_y, 0], color=P.axe, stroke_width=1.5)
        titre_x = libelle("taille du sondage", taille=13, couleur=P.discret).next_to(lab_n, DOWN, buff=0.1)
        legende = VGroup(
            VGroup(Rectangle(width=0.3, height=0.2, stroke_width=0, fill_color=P.accent, fill_opacity=OPACITE_AIRE * 4),
                   libelle("hasard du tirage : ce que mesure la marge", taille=14, couleur=P.texte)).arrange(RIGHT, buff=0.15),
            VGroup(Rectangle(width=0.3, height=0.2, stroke_width=0, fill_color=P.texte_2, fill_opacity=0.85),
                   libelle("erreur observée", taille=14, couleur=P.texte)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        nom = libelle("erreur typique d’un sondage, selon sa taille", taille=15, couleur=P.texte)
        VGroup(nom, legende).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(tete, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(FadeIn(axe), FadeIn(lab_n), FadeIn(titre_x), FadeIn(legende[0]), FadeIn(nom), run_time=0.8)
        self.play(*[FadeIn(b[0], shift=0.1 * UP) for b in barres], run_time=1)
        self.wait(1.5)
        self.play(FadeIn(legende[1]), *[FadeIn(b[1], shift=0.1 * UP) for b in barres], run_time=1.2)
        self.play(FadeIn(rapports, lag_ratio=0.1), run_time=1)
        self.wait(1.5)
        reste = sous_titre("le reste ne vient pas du hasard : il vient de la fabrication du sondage", taille=24)
        reste.next_to(legende, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(reste), run_time=0.6)
        self.wait(2.5)
        self.play(FadeOut(VGroup(axe, lab_n, titre_x, legende, nom, barres, rapports, reste)), run_time=0.8)

        # --- Une critique ancienne : Bourdieu, Dézé -------------------------------------------
        bourdieu = VGroup(
            libelle("pierre bourdieu · « l’opinion publique n’existe pas » · 1973", taille=16, couleur=P.accent),
            VGroup(*[
                sous_titre(t, taille=26)
                for t in (
                    "tout le monde peut avoir une opinion ?",
                    "toutes les opinions se valent ?",
                    "tout le monde s’accorde sur les questions à poser ?",
                )
            ]).arrange(DOWN, aligned_edge=LEFT, buff=0.15),
            citation("un artefact pur et simple", "Les Temps modernes, n° 318, 1973"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        deze = VGroup(
            libelle("alexandre dézé · 10 leçons sur les sondages politiques · 2022", taille=16, couleur=P.accent),
            sous_titre("échantillons par quotas, redressements, formulation des questions", taille=26),
            sous_titre("une « photographie de l’opinion » ?", taille=30, couleur=P.texte),
            libelle("la formule des instituts, mise en question (leçon 3)", taille=13, couleur=P.discret),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        VGroup(bourdieu, deze).arrange(DOWN, aligned_edge=LEFT, buff=0.6).next_to(tete, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(FadeIn(bourdieu[0]), run_time=0.6)
        for ligne in bourdieu[1]:
            self.play(FadeIn(ligne, shift=0.1 * UP), run_time=0.5)
            self.wait(0.3)
        self.play(FadeIn(bourdieu[2]), run_time=0.6)
        self.wait(1.5)
        self.play(FadeIn(deze[0]), FadeIn(deze[1]), run_time=0.8)
        self.wait(1)
        self.play(FadeIn(deze[2]), FadeIn(deze[3]), run_time=0.8)
        self.wait(2)
        self.play(FadeOut(VGroup(bourdieu, deze)), run_time=0.8)

        # --- Les panels en ligne -------------------------------------------------------------
        # L'échange Dézé / Gallard ; la manipulation des panels sera le sujet d'une autre vidéo.
        nom_p = sous_titre("Un angle mort récent : les panels en ligne", taille=34)
        cotes = VGroup(
            citation("inscription […] sans condition et sans contrôle", "Alexandre Dézé, 2022 · inscrit sous une fausse identité"),
            citation("pas impossible, mais […] si décourageante que ce doit être extrêmement rare", "Mathieu Gallard, Ipsos, 2022 · sur l’infiltration des panels"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        limite = libelle("cette étude ne permet ni de détecter ni d’exclure une manipulation", taille=15, couleur=P.discret)
        VGroup(nom_p, cotes, limite).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(tete, DOWN, buff=0.6, aligned_edge=LEFT)
        self.play(FadeIn(nom_p), run_time=0.8)
        for c in cotes:
            self.play(FadeIn(c, shift=0.1 * UP), run_time=0.6)
            self.wait(1.2)
        self.play(FadeIn(limite), run_time=0.6)
        self.wait(2)
        self.play(FadeOut(VGroup(nom_p, cotes, limite)), run_time=0.8)

        # --- Photographie, prédiction, présage --------------------------------------------------
        mots = VGroup(*[titre("", m, "", taille=60) for m in ("photographie", "prédiction", "présage")])
        mots.arrange(RIGHT, buff=1.1).move_to([0, 0.6, 0])
        notes = VGroup(
            libelle("selon les instituts", taille=14, couleur=P.discret),
            libelle("jugée le soir du vote", taille=14, couleur=P.discret),
            libelle("quand l’erreur est commune", taille=14, couleur=P.discret),
        )
        for n, m in zip(notes, mots):
            n.next_to(m, DOWN, buff=0.25)
        gallard = citation("un sondage n’est pas une prédiction", "Mathieu Gallard, Ipsos, 2022", taille=24)
        gallard.next_to(tete, DOWN, buff=0.6, aligned_edge=LEFT)
        mots.shift(DOWN * 1.1)
        notes.shift(DOWN * 1.1)
        self.play(FadeIn(mots[0]), FadeIn(notes[0]), FadeIn(gallard), run_time=0.8)
        self.wait(1.5)
        self.play(FadeIn(mots[1], shift=0.15 * LEFT), FadeIn(notes[1]), run_time=0.8)
        self.play(FadeIn(Cross(mots[0], stroke_color=P.discret, stroke_width=3)), mots[0].animate.set_opacity(0.4), run_time=0.6)
        self.wait(1.5)
        self.play(FadeIn(mots[2], shift=0.15 * LEFT), FadeIn(notes[2]), run_time=0.8)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tete], run_time=0.8)

        # --- Fin -------------------------------------------------------------------------------
        formule = VGroup(
            titre("Une prédiction plus qu’une ", "photographie", ",", taille=46),
            titre("un présage plus qu’une ", "prédiction", ".", taille=46),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        ensemble = sous_titre(
            "quand tous les présages disent la même chose, ils peuvent se tromper ensemble",
            taille=28, couleur=P.texte_2,
        )
        lien = libelle("calculs, données et graphiques interactifs : lien sous la vidéo", taille=16, couleur=P.discret)
        VGroup(formule, ensemble, lien).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to([0, -0.4, 0])
        self.play(FadeIn(formule[0], shift=0.15 * UP), run_time=1)
        self.play(FadeIn(formule[1], shift=0.15 * UP), run_time=1)
        self.wait(1)
        self.play(FadeIn(ensemble), run_time=0.8)
        self.wait(1.5)
        self.play(FadeIn(lien), run_time=0.6)
        self.wait(3)
