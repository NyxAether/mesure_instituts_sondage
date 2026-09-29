"""Séquence 0 — Accroche : primaire de la droite 2016, puis second tour de la présidentielle 2017.

Les trois candidats de la primaire sont d'abord un photomontage à la Karambolage : corps en costume découpés
dans des photos (externe/corps/, outils/decoupe_corps.py), tête en coupure de journal, qui sautillent sans arrêt ;
vient le mème « Quelle indignité ! », joué avec son son (externe/meme/, source dans sources.json) ; puis les
têtes quittent les corps pour le graphe des sondages.

Rendu : .venv/Scripts/python.exe -m manim -ql scenes/s0_accroche.py Accroche
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Line,
    ManimColor,
    Polygon,
    UpdateFromAlphaFunc,
    VGroup,
    Write,
    config,
)
from manimpango import list_fonts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from donnees import TOUS, primaire_2016, sondages_election  # noqa: E402
from theme import SERIF, P, SceneRR, ecrire, fr, libelle, sous_titre, terminal, titre  # noqa: E402

NBSP = " "
PRESIDENTIELLE = ("France", 2017, 2)
CANDIDATS_2017 = {13: "Macron", 3: "Le Pen"}  # identifiants de la base
PORTRAITS = Path(__file__).resolve().parents[1] / "externe" / "portraits"
CORPS = Path(__file__).resolve().parents[1] / "externe" / "corps"
MEME = Path(__file__).resolve().parents[1] / "externe" / "meme" / "quelle_indignite.mp4"
LARGEUR_MEME = 9.6  # à l'écran, dans sa fenêtre terminal (l'extrait est en 16/9)
HAUTEUR_TETE = 1.4  # sur les corps
COU_BAS = 4.6  # du col au bas de la photo : le bas des corps sort du cadre, même quand ils sautent
ECHELLE_CORPS = {"sarkozy": 0.85}  # bras levé et cadrage serré : sans réduction, il paraît plus massif que les autres
PAPIER = ManimColor.from_rgb((240, 234, 220))  # le papier des coupures (outils/decoupe_portraits.py)
HAUTEUR_GRAPHE = 0.8  # sur le graphe, où Juppé et Sarkozy ne sont qu'à 0,8 unité l'un de l'autre
# Police manuscrite de Windows pour les étiquettes écrites à la main ; à défaut, la serif de la charte.
POLICE_ENFANT = "Ink Free" if "Ink Free" in list_fonts() else SERIF


def corps_photo(cle):
    """Corps en costume découpé dans une photo (outils/decoupe_corps.py), réduit (Lanczos) à sa taille à l'écran.

    Renvoie l'image et la position du col dans l'image, en fraction (depuis la gauche, depuis le haut).
    """
    x_cou, y_cou = json.loads((CORPS / "cous.json").read_text(encoding="utf-8"))[cle]
    png = Image.open(CORPS / f"{cle}_decoupe.png")
    hauteur = COU_BAS * ECHELLE_CORPS.get(cle, 1) / (1 - y_cou)
    pixels = round(hauteur / config.frame_height * config.pixel_height)
    png = png.resize((round(png.width * pixels / png.height), pixels), Image.Resampling.LANCZOS)
    return ImageMobject(np.array(png)).set(height=hauteur), (x_cou, y_cou)


def papier_decoupe(points, graine, couleur=PAPIER):
    """Morceau de papier coupé aux ciseaux : polygone aux sommets un peu déplacés, bord discret de la charte."""
    rng = np.random.default_rng(graine)
    sommets = [np.array([x + rng.normal(0, 0.02), y + rng.normal(0, 0.02), 0]) for x, y in points]
    return Polygon(
        *sommets, fill_color=couleur, fill_opacity=1,
        stroke_color=P.ligne_forte[0], stroke_opacity=P.ligne_forte[1], stroke_width=1.2,
    )


def sautiller(groupe, graine, bascule_max=0.14):
    """Fait sautiller le groupe sans arrêt, à son propre rythme : hauteurs, durées et pauses tirées au hasard,
    petits pas de côté, bascule en l'air. Chaque bonhomme a sa graine, donc ils ne sont jamais synchronisés.

    Renvoie la fonction qui arrête les sauts et repose le groupe droit.
    """
    rng = np.random.default_rng(graine)
    pivot = groupe.get_bottom()
    etat = {"t": 0.0, "saut": None, "prochain": rng.uniform(0, 0.3), "x": 0.0, "decalage": np.zeros(3), "angle": 0.0}

    def placer(m, decalage, angle):
        m.rotate(-etat["angle"], about_point=pivot + etat["decalage"])
        m.shift(decalage - etat["decalage"])
        m.rotate(angle, about_point=pivot + decalage)
        etat.update(decalage=decalage, angle=angle)

    def maj(m, dt):
        etat["t"] += dt
        s = etat["t"]
        if etat["saut"] is None and s >= etat["prochain"]:
            dx = float(np.clip(rng.normal(0, 0.1) - 0.5 * etat["x"], -0.18, 0.18))  # rappel vers la place de départ
            etat["saut"] = (etat["prochain"], rng.uniform(0.25, 0.5), rng.uniform(0.08, 0.3), dx, rng.uniform(-bascule_max, bascule_max))
        decalage, angle = RIGHT * etat["x"], 0.0
        if etat["saut"] is not None:
            debut, d, h, dx, bascule = etat["saut"]
            u = min((s - debut) / d, 1)
            decalage = decalage + RIGHT * dx * u + UP * h * np.sin(np.pi * u)
            angle = bascule * np.sin(np.pi * u)
            if u >= 1:
                etat["x"] += dx
                etat["saut"] = None
                etat["prochain"] = s + rng.uniform(0, 0.15)
        placer(m, decalage, angle)

    def arreter():
        groupe.remove_updater(maj)
        placer(groupe, RIGHT * etat["x"], 0.0)

    groupe.add_updater(maj)
    return arreter


def jouer_meme(scene):
    """Joue l'extrait image par image, à la cadence du rendu, avec son son, dans une fenêtre terminal de la charte.

    ffmpeg en tire les images (à leur taille à l'écran) et le son dans le dossier media/, ignoré par git.
    """
    hauteur = LARGEUR_MEME * 9 / 16
    pixels = 2 * round(hauteur / config.frame_height * config.pixel_height / 2)  # ffmpeg veut une hauteur paire
    cache = Path(config.media_dir) / "meme"
    cache.mkdir(parents=True, exist_ok=True)
    fps = config.frame_rate
    son = cache / "quelle_indignite.wav"
    prefixe = f"image_{pixels}p{fps:g}_"
    for vieux in cache.glob(prefixe + "*.png"):
        vieux.unlink()
    ffmpeg = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(MEME)]
    subprocess.run([*ffmpeg, "-vn", str(son)], check=True)
    subprocess.run([
        *ffmpeg, "-vf", f"fps={fps},scale=-2:{pixels}:flags=lanczos", str(cache / (prefixe + "%04d.png")),
    ], check=True)
    images = [np.array(Image.open(f).convert("RGBA")) for f in sorted(cache.glob(prefixe + "*.png"))]

    marge = 0.16
    fenetre, centre = terminal(
        LARGEUR_MEME + 2 * marge, hauteur + 2 * marge, "play quelle_indignite.mp4",
        statut="france 2 · 17 nov. 2016", dossier="~/sondages",
    )
    VGroup(fenetre).move_to(ORIGIN)
    centre = fenetre[0].get_bottom() + UP * (hauteur / 2 + marge)
    ecran = ImageMobject(images[0]).set(height=hauteur).move_to(centre)
    horloge = {"t": 0.0}

    def defiler(m, dt):
        horloge["t"] += dt
        m.pixel_array = images[min(int(horloge["t"] * fps), len(images) - 1)]

    scene.play(FadeIn(fenetre), run_time=0.3)
    scene.add(ecran)
    scene.add_sound(str(son))
    ecran.add_updater(defiler)
    scene.wait(len(images) / fps)
    ecran.clear_updaters()
    scene.remove(ecran, fenetre)


def tete(cle, inclinaison, hauteur=HAUTEUR_TETE):
    """Portrait découpé façon coupure de journal (voir outils/decoupe_portraits.py).

    Manim réduit les images sans les filtrer, ce qui fait moirer la trame : on les réduit donc
    d'abord (Lanczos) à leur taille à l'écran.
    """
    png = Image.open(PORTRAITS / f"{cle}_decoupe.png")
    pixels = round(hauteur / config.frame_height * config.pixel_height)
    png = png.resize((round(png.width * pixels / png.height), pixels), Image.Resampling.LANCZOS)
    image = ImageMobject(np.array(png)).set(height=hauteur)
    image.inclinaison = inclinaison
    image.set_z_index(10)  # toujours devant les courbes et les points, ajoutés après elles
    return image.rotate(inclinaison)


def trajet(image, position, dandinement=0.07, oscillations=3, echelle=1):
    """Déplace une tête le long de position(alpha), en la faisant tanguer comme un papier qu'on promène.

    echelle : facteur de taille atteint à l'arrivée.
    """
    def pas(m, alpha):
        angle = m.inclinaison + dandinement * np.sin(alpha * oscillations * 2 * np.pi)
        taille = 1 + (echelle - 1) * alpha
        m.rotate(angle - m.angle_courant).scale(taille / m.taille_courante).move_to(position(alpha))
        m.angle_courant, m.taille_courante = angle, taille
    image.angle_courant, image.taille_courante = image.inclinaison, 1
    return UpdateFromAlphaFunc(image, pas)


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

        courbes, noms, sommets = VGroup(), VGroup(), []
        for cle, (nom, couleur) in candidats.items():
            points = [np.array([x(i), y(s[cle]), 0]) for i, s in enumerate(sondages)]
            sommets.append(points)
            ligne = VGroup(*[Line(a, b, color=couleur, stroke_width=2.5) for a, b in zip(points, points[1:])])
            dots = VGroup(*[Dot(q, radius=0.06, color=couleur) for q in points])
            courbes.add(VGroup(ligne, dots))
            noms.add(libelle(nom.lower(), taille=14, couleur=couleur).next_to(points[0], DOWN if cle == "fillon" else UP, buff=0.15))

        inclinaisons = {"fillon": -0.12, "juppe": 0.08, "sarkozy": -0.03}
        decalage_tas = {"fillon": -0.45, "juppe": 0.05, "sarkozy": 0.5}
        tetes = {cle: tete(cle, inclinaisons[cle]) for cle in candidats}
        dessus = UP * (HAUTEUR_GRAPHE / 2 + 0.1)

        # --- Les trois candidats, en photomontage -------------------------------------------------
        # Dans l'ordre où le narrateur les nomme : Juppé, Sarkozy, Fillon.
        ordre = {"juppe": (-4.5, "Alain"), "sarkozy": (0, "Nicolas"), "fillon": (4.6, "François")}
        corps, prenoms, candidats_photo = {}, Group(), {}
        for i, (cle, (x_c, prenom)) in enumerate(ordre.items()):
            image, (fx, fy) = corps_photo(cle)
            # Col placé pour que le bas de la photo reste sous le cadre (−4), avec la marge d'un saut
            y_cou = -4.3 + COU_BAS * ECHELLE_CORPS.get(cle, 1)
            # Le col de la photo vient au point (x_c, y_cou)
            image.move_to([x_c + (0.5 - fx) * image.width, y_cou + (fy - 0.5) * image.height, 0])
            corps[cle] = image
            tetes[cle].move_to([x_c, y_cou + HAUTEUR_TETE / 2 - 0.2, 0])
            couleur = candidats[cle][1]
            if cle == "sarkozy":  # le gris du graphe est trop pâle pour une écriture
                couleur = couleur.interpolate(P.texte, 0.35)
            mot = ecrire(prenom, POLICE_ENFANT, 30, couleur)
            l, r, b, h = mot.get_left()[0] - 0.2, mot.get_right()[0] + 0.2, mot.get_bottom()[1] - 0.1, mot.get_top()[1] + 0.12
            etiquette = VGroup(papier_decoupe([(l, b), (l, h), (r, h), (r, b)], graine=20 + i), mot)
            prenoms.add(etiquette.move_to([x_c, -3.45, 0]).rotate(0.05 * (-1) ** i))
            candidats_photo[cle] = Group(corps[cle], tetes[cle])

        self.play(*[FadeIn(corps[cle], shift=UP * 1.5) for cle in ordre], run_time=0.9)
        self.play(*[FadeIn(tetes[cle], scale=1.4) for cle in ordre], FadeIn(prenoms, lag_ratio=0.3), run_time=0.8)
        self.add(*candidats_photo.values())
        self.add(prenoms)  # les étiquettes restent devant les corps
        arrets = [sautiller(groupe, graine=100 + i, bascule_max=0.04) for i, groupe in enumerate(candidats_photo.values())]
        self.wait(4.5)

        # La banderole de la primaire : un ruban de papier découpé, pans fourchus
        mot = titre("Primaire de la ", "droite", " et du centre", taille=40)
        mot.move_to([0, 3.05, 0])
        l, r = mot.get_left()[0] - 0.4, mot.get_right()[0] + 0.4
        b, h = mot.get_bottom()[1] - 0.22, mot.get_top()[1] + 0.22
        m = (b + h) / 2
        pans = Group(
            papier_decoupe([(l + 0.3, h - 0.25), (l - 0.9, h - 0.25), (l - 0.55, m - 0.18), (l - 0.9, b - 0.18), (l + 0.3, b - 0.18)], 31),
            papier_decoupe([(r - 0.3, h - 0.25), (r + 0.9, h - 0.25), (r + 0.55, m - 0.18), (r + 0.9, b - 0.18), (r - 0.3, b - 0.18)], 32),
        )
        ruban = papier_decoupe([(l, b), (l, h), (r, h + 0.03), (r, b - 0.02)], 33)
        banderole = Group(pans, ruban, mot)
        self.play(FadeIn(Group(pans, ruban), shift=DOWN * 0.8), run_time=0.7)
        self.play(Write(mot), run_time=1.5)
        self.wait(2.5)

        # Le mème « Quelle indignité ! », en coupe franche, avec son son (le narrateur se tait)
        dessin = [*candidats_photo.values(), prenoms, pans, ruban, mot]  # tels qu'ajoutés à la scène
        self.remove(*dessin)
        jouer_meme(self)
        self.add(*dessin)
        self.wait(1.5)
        for arreter in arrets:
            arreter()

        # --- Les têtes quittent les corps pour le graphe des sondages ------------------------------
        # Chacune se pose à gauche de l'axe, en face de son score dans le premier sondage (Harris, 7 au 9 novembre).
        envols = []
        for cle, points in zip(candidats, sommets):
            depart, arrivee = tetes[cle].get_center(), np.array([gauche - 1.3, points[0][1], 0])
            envols.append(trajet(
                tetes[cle], lambda a, d=depart, r=arrivee: d + (r - d) * a + UP * 0.8 * np.sin(np.pi * a),
                echelle=HAUTEUR_GRAPHE / HAUTEUR_TETE,
            ))
        self.play(
            FadeOut(Group(prenoms, banderole, *corps.values())), *envols,
            FadeIn(tag), FadeIn(grille), FadeIn(lab_y), FadeIn(lab_x, lag_ratio=0.1), FadeIn(source),
            run_time=1.8,
        )
        premier = sondages[0]
        fillon_debut = sous_titre(f"Fillon troisième, {pour_cent(premier['fillon'])}", taille=26, couleur=P.series[0])
        fillon_debut.next_to(noms[0], DOWN, buff=0.2).align_to([gauche, 0, 0], LEFT)
        self.play(FadeIn(noms), FadeIn(VGroup(*[dots[0] for _, dots in courbes])), run_time=0.6)
        self.play(FadeIn(fillon_debut), run_time=0.6)
        self.wait(4)

        # Chaque tête rejoint la pointe de sa courbe et la suit, un peu au-dessus ; elles finissent en tas sur le dernier sondage.
        def pointe(points, alpha):
            # Create trace les segments l'un après l'autre, en temps égal (lag_ratio = 1).
            u = alpha * (len(points) - 1)
            i = min(int(u), len(points) - 2)
            return points[i] + (u - i) * (points[i + 1] - points[i])

        suivis = []
        for (ligne, dots), cle, points in zip(courbes, candidats, sommets):
            debut, fin = tetes[cle].get_center(), points[-1] + dessus + RIGHT * decalage_tas[cle]
            suivis += [
                FadeIn(dots[1:], lag_ratio=0.1), Create(ligne),
                trajet(tetes[cle], lambda a, p=points, d=debut, f=fin: (
                    pointe(p, a) + dessus + (d - p[0] - dessus) * (1 - a) ** 3 + (f - p[-1] - dessus) * a ** 3
                )),
            ]
        self.play(*suivis, run_time=3.5)
        self.wait(2)

        # Le résultat du premier tour
        dernier = len(sondages) - 1
        x_tetes = x_res + 0.62  # les têtes atterrissent juste à droite du résultat, les valeurs après elles
        sauts, points_res, lab_valeurs, ecarts = VGroup(), VGroup(), VGroup(), VGroup()
        for cle, (nom, couleur) in candidats.items():
            depart = np.array([x(dernier), y(sondages[dernier][cle]), 0])
            arrivee = np.array([x_res, y(resultat[cle]), 0])
            sauts.add(DashedLine(depart, arrivee, color=couleur, dash_length=0.08, stroke_width=2))
            points_res.add(Dot(arrivee, radius=0.11, color=couleur))
            lab_valeurs.add(libelle(pour_cent(resultat[cle]), taille=15, couleur=couleur).next_to(arrivee, RIGHT, buff=1.15))
            ecart = resultat[cle] - sondages[dernier][cle]
            ecarts.add(libelle(f"{signe(ecart)} pts", taille=13, couleur=couleur).next_to(lab_valeurs[-1], DOWN, buff=0.06, aligned_edge=LEFT))
        envols = []
        for cle in candidats:
            image = tetes[cle]
            depart, arrivee = image.get_center(), np.array([x_tetes, y(resultat[cle]), 0])
            # Petite parabole : la tête décolle avant de filer vers son résultat.
            envols.append(trajet(
                image, lambda a, d=depart, r=arrivee: d + (r - d) * a + UP * 0.6 * np.sin(np.pi * a),
                dandinement=0.1, oscillations=2,
            ))
        self.play(FadeIn(lab_res), *[Create(s) for s in sauts], *envols, run_time=1.6)
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
        # Chaque tête entre par la gauche et se pose devant sa bande, juste avant ses points.
        for (cand, cle), groupe, inclinaison in zip({13: "macron", 3: "lepen"}.items(), dots_2017, (0.07, -0.08)):
            b, h = panneaux[cand]
            image = tete(cle, inclinaison, hauteur=1.3)
            depart, arrivee = np.array([-8.2, (b + h) / 2 + 0.4, 0]), np.array([-6.2, (b + h) / 2, 0])
            image.move_to(depart)
            self.add(image)
            self.play(trajet(image, lambda a, d=depart, r=arrivee: d + (r - d) * a + UP * 0.35 * np.sin(np.pi * a)), run_time=0.7)
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
