# Vidéo « Ce que valent vraiment les sondages » en Remotion

La vidéo est écrite avec [Remotion](https://www.remotion.dev/) (React, rendu par Chromium puis ffmpeg). Elle reprend le plan et le minutage de la version Manim d'origine, retirée du dépôt après le portage (dernière version : `video/scenes/` au commit `1ea94a2`). Le bilan du premier essai (séquence 0) est dans [COMPARAISON.md](COMPARAISON.md).

## Rendu

Il faut Node 20 ou plus récent. Depuis `video-js/` :

```sh
npm install
npm run render      # → out/video.mp4 : les sept séquences bout à bout, 1080p60
npm run studio      # aperçu interactif dans le navigateur, avec timeline
```

Une séquence seule : `npx remotion render Sequence3 out/sequence3.mp4`. Une seule image : `npx remotion still Sequence0 out/image.png --frame=630`. Thème sombre : `REMOTION_THEME=dark`.

`npm run render` et `npm run studio` lancent d'abord [ressources.mjs](ressources.mjs). Ce script :
- copie dans `public/` (ignoré par git) les têtes, les corps, le mème et les polices, pris dans `video/externe/` et `rr-fonts/` ;
- tire la couleur du papier des coupures de `video/outils/decoupe_portraits.py`.

Rien n'est donc dupliqué à la main.

## Données

Aucun chiffre n'est écrit dans le code. Chaque séquence lit `donnees/sN.json`, produit par `export/exporter_sN.py` à partir de `video/donnees.py` (base des sondages, tirages numpy avec graine). Depuis la racine du dépôt :

```sh
video/.venv/Scripts/python.exe video-js/export/exporter_s0.py
```

## Organisation

| Dossier ou fichier | Rôle |
|---|---|
| `src/sequences.ts` | liste ordonnée des séquences ; chacune est une composition (`Sequence0` … `Sequence6`) et une partie de la composition `Video` |
| `src/sequences/sN_*/` | une séquence : composants et `temps.ts` (minutage plan par plan, en constantes nommées, pour pouvoir le caler plus tard sur la voix) |
| `src/lib/charte.ts` | couleurs, polices, rayon et courbe d'animation lus dans `video/rr-tokens.json` |
| `src/lib/outils.tsx` | repère Manim (8 unités de haut, 135 px par unité), avancement et échelonnement, hasard reproductible, nombres à la française, texte positionné |
| `src/lib/composants.tsx` | en-tête de section, titre à un mot italique, curseur |
| `src/lib/axes.ts` | échelles des graphes (d3-scale) |
| `src/lib/papier.tsx` | papier découpé (étiquettes, banderole), fenêtre terminal de la charte |
| `src/lib/sautiller.ts` | programme des sauts, tiré d'avance avec une graine par candidat |
| `export/`, `donnees/` | scripts Python d'export et JSON générés |

L'étiquette manuscrite de la séquence 0 utilise la police Ink Free, présente sur Windows. Ailleurs, elle se rabat sur Newsreader.
