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
| [donnees.py](donnees.py) | chargement de `docs/data/erreurs.js` (`TOUS`, `FRANCE`) |
| `scenes/sN_*.py` | une scène Manim par séquence |

Rendu (depuis `video/`) :

```sh
uv sync
.venv/Scripts/python.exe -m manim -ql scenes/s1_promesse.py Promesse            # thème sombre
RR_THEME=light .venv/Scripts/python.exe -m manim -ql -o Promesse_clair scenes/s1_promesse.py Promesse
```

Les vidéos sortent dans `media/` (ignoré par git).

### Choix de style retenus

- **Polices :** titres et annotations en Newsreader (`titre`, `sous_titre`), libellés en JetBrains Mono minuscule (`libelle`). Inter détonnait à côté des titres et n'est plus utilisé dans les scènes. Newsreader n'a pas de glyphe « → » : l'éviter dans `sous_titre`.
- **Rendu du texte :** tout texte passe par `ecrire` / `TexteNet`. Il est rendu 10 fois plus grand puis réduit, sinon Pango arrondit la position des lettres et la chasse devient irrégulière. Le canevas est très large pour qu'il n'y ait pas de retour à la ligne automatique.
- **Couleurs :** vote a en prune (série 1), vote b en bleu (série 2). Les éléments qui représentent un tirage prennent la couleur du vote majoritaire de l'échantillon. La marge à 95 % est une zone accent à 15 % d'opacité.
- **Formules :** `MathTex` (LaTeX, Computer Modern).
- **Thème :** sombre par défaut. Le clair a été rendu pour comparaison ; le choix reste ouvert.

## Avancement par séquence

| # | Séquence | État |
|---|---|---|
| 0 | Accroche : primaire 2016, présidentielle 2017 (2d tour) | à faire |
| 1 | La promesse : ce que veut dire « ± 3 points » | **faite**, relue et validée en 480p ([scenes/s1_promesse.py](scenes/s1_promesse.py)) |
| 2 | L'entonnoir : 45 % hors marge | brouillon écrit avant le feu vert, **non relu** ([scenes/s2_entonnoir.py](scenes/s2_entonnoir.py)) : à reprendre ou à jeter quand on l'ouvrira |
| 3 | L'erreur ne baisse pas avec la taille | à faire |
| 4 | Taille équivalente | à faire |
| 5 | Erreur partagée, mimétisme | à faire |
| 6 | Une prédiction plus qu'une photographie, un présage plus qu'une prédiction | à faire ; paragraphe sur les panels à compléter avec les travaux de Romain |

Séquence 1, déroulé : population de points, 3 tirages lents (résultat affiché, les personnes tirées rejoignent l'axe), tirages 4 à 40 de plus en plus rapides, fusion des traits en barres, histogramme jusqu'à 600 tirages, bande à 95 % (± 3,1 points), formule, passage à n = 4 000 (± 1,5 point), puis l'écran « ± 3 points, 19 fois sur 20 » suivi de « 1 fois sur 20 : plus de 3 points d'écart » et `> vérifions-la`.

## Reste à faire

1. Séquences 0 et 2 à 6, une par une.
2. Choix définitif entre thème sombre et thème clair.
3. Formules : essayer XeLaTeX + `fontspec` avec Newsreader (et une police mathématique proche, par exemple Libertinus Math) pour harmoniser les formules avec les titres.
4. Voix : enregistrement, puis calage des `self.wait()` sur les durées réelles (ou `manim-voiceover`).
5. Montage : concaténation ffmpeg des séquences, piste voix, sous-titres `.srt` issus du script, export 1080p.
6. Vérification des chiffres affichés contre la page `docs/erreurs.html`, et recontrôle ligne à ligne des chiffres de la primaire 2016 sur la page Wikipédia citée dans le script.
