# Vidéo « Ce que valent vraiment les sondages » : plan et avancement

Vidéo explicative de 9 à 10 minutes tirée de [docs/erreurs.html](../docs/erreurs.html). Les animations sont générées par code (Manim) à partir des données exportées, pour que chaque chiffre à l'écran soit exact et régénérable. Le texte de la voix off est dans [script.md](script.md).

## Méthode de travail

- **Une séquence à la fois.** Chaque séquence est proposée pour review ; on ne passe à la suivante qu'avec un feu vert explicite.
- **Les ajustements transverses se font à la fin** (voir « Reste à faire »).
- **Aucun chiffre en dur :** tout vient de [donnees.py](donnees.py), qui lit `docs/data/erreurs.js`. Les données hors base (primaire 2016) iront dans un fichier séparé, avec leur source.

## Mise en place

| Fichier | Rôle |
|---|---|
| [pyproject.toml](pyproject.toml) | projet uv isolé du projet Poetry, avec Manim 0.21 |
| [rr-tokens.json](rr-tokens.json) | tokens de la charte rr/, tirés par `rr-design pull` (déclarés dans `rr-design.toml`) : ne pas modifier |
| [theme.py](theme.py) | couleurs lues dans les tokens, polices de `rr-fonts/`, helpers `titre`, `sous_titre`, `libelle`, `entete`, `fr` |
| [donnees.py](donnees.py) | chargement de `docs/data/erreurs.js` (`TOUS`, `FRANCE`) ; `glissante_equivalents()` lit `mesure_erreurs/bss.p` (pandas) pour les courbes de taille équivalente selon la taille réelle |
| `scenes/sN_*.py` | une scène Manim par séquence |

Rendu (depuis `video/`) :

```sh
uv sync
.venv/Scripts/python.exe -m manim -ql scenes/s1_theorie.py Theorie            # thème clair
RR_THEME=dark .venv/Scripts/python.exe -m manim -ql -o Theorie_sombre scenes/s1_theorie.py Theorie
```

Les vidéos sortent dans `media/` (ignoré par git).

### Choix de style retenus

- **Polices :** titres et annotations en Newsreader (`titre`, `sous_titre`), libellés en JetBrains Mono minuscule (`libelle`). Inter détonnait à côté des titres et n'est plus utilisé dans les scènes. Newsreader n'a pas de glyphe « → » : l'éviter dans `sous_titre`.
- **Rendu du texte :** tout texte passe par `ecrire` / `TexteNet`. Il est rendu 10 fois plus grand puis réduit, sinon Pango arrondit la position des lettres et la chasse devient irrégulière. Le canevas est très large pour qu'il n'y ait pas de retour à la ligne automatique.
- **Couleurs :** vote a en prune (série 1), vote b en bleu (série 2). Les éléments qui représentent un tirage prennent la couleur du vote majoritaire de l'échantillon. La marge à 95 % est une zone accent à 15 % d'opacité.
- **Formules :** `MathTex` (LaTeX, Computer Modern).
- **Thème :** clair, retenu et mis par défaut. Le sombre reste disponible avec `RR_THEME=dark`.
- **Simplifier :** la vidéo vise la compréhension, les subtilités sont laissées à `docs/erreurs.html`. Les chiffres affichés sont toujours les vrais ; seule la visualisation simplifie. Séquence 2 : points colorés selon l'entonnoir à p = 50 %, alors que le pourcentage affiché (45 %) et la marge de l'exemple (± 2,3) sont les vrais, calculés avec la marge propre à chaque parti.
- **Ton :** les écarts sont rapportés à la théorie (« attendu en théorie »), jamais à une « promesse » des instituts.

## Avancement par séquence

| # | Séquence | État |
|---|---|---|
| 0 | Accroche : primaire 2016, présidentielle 2017 (2d tour) | à faire |
| 1 | La théorie : ce que veut dire « ± 3 points » | **faite**, relue et validée en 480p ([scenes/s1_theorie.py](scenes/s1_theorie.py)) ; titre et fin reformulés (« prévoit », « vérifions ») |
| 2 | L'entonnoir : 45 % hors marge | **faite**, validée en 480p ([scenes/s2_entonnoir.py](scenes/s2_entonnoir.py)) |
| 3 | L'excédent ne diminue pas avec la taille | **faite**, validée en 480p ([scenes/s3_taille.py](scenes/s3_taille.py)) ; zoom animé de l'axe vertical (bornes en `ValueTracker`) |
| 4 | Taille équivalente | **faite**, validée en 480p ([scenes/s4_equivalente.py](scenes/s4_equivalente.py)) : l'entonnoir de la séquence 2 s'élargit jusqu'à contenir 95 % des écarts (÷ 11, 184 personnes), puis la mesure de l'étude (222, témoin 1 973), courbes selon la taille réelle, boîtes selon les jours |
| 5 | Erreur partagée, mimétisme | **en relecture** ([scenes/s5_partage.py](scenes/s5_partage.py)) : Royaume-Uni 2015 (tirages simulés puis vrais sondages, moyenne à côté), essaims consensus (80 %) et resserrement (3 %), explications des instituts et des chercheurs en deux colonnes ; Venezuela 2013 retiré (données douteuses) |
| 6 | Une prédiction plus qu'une photographie, un présage plus qu'une prédiction | **brouillon, à reprendre** (ne convainc pas encore) ([scenes/s6_presage.py](scenes/s6_presage.py)) : précautions (barre France 3 %), erreur attendue et observée par taille (×2,3 à ×4,8), Bourdieu et Dézé, panels (Dézé contre Gallard ; leur manipulation sera le sujet d'une autre vidéo), photographie, prédiction, présage |

Séquence 1, déroulé : population de points, 3 tirages lents (résultat affiché, les personnes tirées rejoignent l'axe), tirages 4 à 40 de plus en plus rapides, fusion des traits en barres, histogramme jusqu'à 600 tirages, bande à 95 % (± 3,1 points), formule, passage à n = 4 000 (± 1,5 point), puis l'écran « ± 3 points, 19 fois sur 20 » suivi de « 1 fois sur 20 : plus de 3 points d'écart » et `> vérifions`.

## Reste à faire

0. Page : la médiane glissante de la taille équivalente (`analyses/erreurs.py`) dépend de l'ordre des sondages de même taille après tri, donc des versions de pandas et numpy (écarts jusqu'à ± 130 sur la médiane). À rendre déterministe (tri par taille puis identifiant) ou à remplacer par un lissage en taille.

1. Séquences 0, 5 et 6, une par une.
2. Formules : essayer XeLaTeX + `fontspec` avec Newsreader (et une police mathématique proche, par exemple Libertinus Math) pour harmoniser les formules avec les titres.
3. Voix : enregistrement, puis calage des `self.wait()` sur les durées réelles (ou `manim-voiceover`).
4. Montage : concaténation ffmpeg des séquences, piste voix, sous-titres `.srt` issus du script, export 1080p.
5. Vérification des chiffres affichés contre la page `docs/erreurs.html`, et recontrôle ligne à ligne des chiffres de la primaire 2016 sur la page Wikipédia citée dans le script.
