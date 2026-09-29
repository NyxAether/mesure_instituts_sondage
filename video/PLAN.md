# Vidéo « Ce que valent vraiment les sondages » : plan et avancement

Vidéo explicative de 9 à 10 minutes tirée de [docs/erreurs.html](../docs/erreurs.html). Les animations sont générées par code (Remotion : React, TypeScript) à partir de données exportées, pour que chaque chiffre à l'écran soit exact et régénérable. Le texte de la voix off est dans [script.md](script.md).

La vidéo a d'abord été écrite en Manim (Python). Elle a été portée en Remotion dans [../video-js/](../video-js/), qui est désormais le projet vidéo ; les scènes Manim ont été retirées du dépôt (dernière version au commit `1ea94a2`, dossier `video/scenes/`). Le bilan du premier essai est dans [../video-js/COMPARAISON.md](../video-js/COMPARAISON.md).

## Organisation

| Où | Rôle |
|---|---|
| [../video-js/](../video-js/) | le projet Remotion : une composition par séquence (`Sequence0` … `Sequence6`) et `Video` qui les enchaîne ; voir son [README](../video-js/README.md) |
| [donnees.py](donnees.py) | chargement de `docs/data/erreurs.js` (`TOUS`, `FRANCE`) ; `glissante_equivalents()` lit `mesure_erreurs/bss.p` (pandas) pour les courbes de taille équivalente selon la taille réelle |
| `../video-js/export/exporter_sN.py` | un export par séquence, qui écrit `../video-js/donnees/sN.json` (y compris les tirages numpy avec graine) |
| [rr-tokens.json](rr-tokens.json) | tokens de la charte rr/, tirés par `rr-design pull` (déclarés dans `rr-design.toml`) : ne pas modifier |
| [externe/](externe/) | ressources hors base, avec leur source : `primaire_2016.json` (Wikipédia, révision notée), portraits et corps découpés, mème |
| [outils/](outils/) | détourage des portraits et des corps (`uv run --script`) |
| [pyproject.toml](pyproject.toml) | projet uv des données Python (pandas) |

Rendu (depuis `video-js/`) : `npm run render` pour la vidéo entière, `npm run studio` pour l'aperçu. Régénérer les chiffres : `video/.venv/Scripts/python.exe video-js/export/exporter_sN.py` depuis la racine du dépôt.

## Méthode de travail

- **Aucun chiffre en dur :** tout vient de [donnees.py](donnees.py) via les JSON exportés. Les données hors base (primaire 2016) sont dans un fichier séparé, avec leur source.
- **Tout est une fonction du numéro d'image** : pas d'état d'une image à l'autre. Le minutage de chaque séquence est dans son `temps.ts`, en constantes nommées, pour pouvoir le caler plus tard sur la voix.
- **Les ajustements transverses se font à la fin** (voir « Reste à faire »).

### Choix de style retenus

- **Polices :** titres et annotations en Newsreader, libellés en JetBrains Mono minuscule. Newsreader n'a pas de glyphe « → » : l'éviter.
- **Couleurs :** vote a en prune (série 1), vote b en bleu (série 2). Les éléments qui représentent un tirage prennent la couleur du vote majoritaire de l'échantillon. La marge à 95 % est une zone accent à 15 % d'opacité.
- **Formules :** KaTeX (séquence 1). Les mettre en Newsreader tient à `HARMONISER_NEWSREADER` dans `Formule.tsx`.
- **Thème :** clair par défaut ; le sombre existe (`REMOTION_THEME=dark`) mais n'a pas été relu.
- **Simplifier :** la vidéo vise la compréhension, les subtilités sont laissées à `docs/erreurs.html`. Les chiffres affichés sont toujours les vrais ; seule la visualisation simplifie. Séquence 2 : points colorés selon l'entonnoir à p = 50 %, alors que le pourcentage affiché (45 %) et la marge de l'exemple (± 2,3) sont les vrais, calculés avec la marge propre à chaque parti.
- **Ton :** les écarts sont rapportés à la théorie (« attendu en théorie »), jamais à une « promesse » des instituts.

## Avancement par séquence

Toutes les séquences sont portées en Remotion (durée totale 349,5 s). Le portage a été contrôlé sur des images comparées au rendu Manim, pas encore relu par séquence.

| # | Séquence | État |
|---|---|---|
| 0 | Accroche : primaire 2016, présidentielle 2017 (2d tour) | portée, **en relecture** : photomontage, mème avec son, courbes de la primaire, 2017 en deux bandes, titre. La légende « chaque bande : 6 points de part et d'autre du résultat » est à reformuler (des points dépassent) |
| 1 | La théorie : ce que veut dire « ± 3 points » | portée (60,1 s) ; tirages numpy exportés à l'identique |
| 2 | L'entonnoir : 45 % hors marge | portée (27,7 s) ; composant d'entonnoir réutilisable |
| 3 | L'excédent ne diminue pas avec la taille | portée (39,1 s) ; zoom de l'axe vertical par bornes interpolées |
| 4 | Taille équivalente | portée (52,1 s) ; a sa propre copie de l'entonnoir, à fusionner avec celle de la séquence 2 |
| 5 | Erreur partagée, mimétisme | portée (64,0 s), **en relecture** |
| 6 | Une prédiction plus qu'une photographie, un présage plus qu'une prédiction | portée telle quelle (55,6 s), **brouillon, à reprendre** (ne convainc pas encore) : quatre blocs de texte pour un seul graphique, précautions en tête, aire « hasard » trop petite, pas d'unité, triade sans image ; les panels (Dézé contre Gallard) restent, leur manipulation sera le sujet d'une autre vidéo |

## Reste à faire

0. Page : la médiane glissante de la taille équivalente (`analyses/erreurs.py`) dépend de l'ordre des sondages de même taille après tri, donc des versions de pandas et numpy (écarts jusqu'à ± 130 sur la médiane). À rendre déterministe (tri par taille puis identifiant) ou à remplacer par un lissage en taille.
1. Relire les séquences portées, une par une ; reprendre la séquence 6.
2. Fusionner l'entonnoir de la séquence 4 avec celui de la séquence 2.
3. Voix : enregistrement, puis calage des durées de `temps.ts` sur les fichiers réels (`<Audio>` par réplique et `calculateMetadata`).
4. Montage : piste voix, sous-titres `.srt` issus du script, export 1080p.
5. Vérification des chiffres affichés contre la page `docs/erreurs.html`, et recontrôle ligne à ligne des chiffres de la primaire 2016 sur la page Wikipédia citée dans le script.
