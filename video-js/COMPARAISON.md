# Séquence 0 : Remotion face à Manim

Même séquence, même minutage (50,95 s), mêmes ressources, mêmes chiffres. Version Manim : [video/scenes/s0_accroche.py](../video/scenes/s0_accroche.py). Version Remotion : [src/](src/), rendu dans `out/sequence0.mp4`.

## Pourquoi Remotion

Motion Canvas était l'autre candidat. Remotion l'a emporté sur trois points :
- **le mème avec son son** : `<OffthreadVideo>` le joue, et le son est mixé au rendu ; Motion Canvas gère mal l'audio d'un clip intégré ;
- **la charte** : elle se lit telle quelle, avec les tokens JSON importés et le terminal écrit en CSS comme dans `rr-components.css` ;
- **le rendu** : une seule commande.

## Qu'est-ce qui a été plus simple ?

- **Le texte.** Chromium compose Newsreader et JetBrains Mono correctement, sans rien de plus. Le mot en italique prune est un `<span>`. Manim, lui, a besoin de tout le détour `TexteNet` / `SURECHELLE` : le texte y est rendu dix fois trop grand puis réduit, sur un canevas géant, pour éviter une chasse irrégulière.
- **Les images.** Les têtes et les corps se posent à leur taille finale, sans moirage de la trame. En Manim, il fallait les réduire d'abord avec Pillow (Lanczos).
- **Le mème.** Une balise suffit, et le son suit. En Manim, ffmpeg extrait d'abord chaque image et le son dans un cache, puis un updater fait défiler les images.
- **La charte.** `rr-tokens.json` est importé directement. Le terminal reprend presque ligne pour ligne le CSS de `rr-components.css` : rayon, bordure 1 px, filet tireté, point sauge. Le gris plus soutenu de Sarkozy est un `color-mix()` de deux tokens. Même la courbe d'animation vient des tokens (`motion.ease`).
- **Le temps de rendu.** 1 min 17 pour la séquence en 1080p60, contre 7 min 35 pour Manim, sur la même machine. Remotion rend 8 images en parallèle.
- **L'aperçu.** `npm run studio` ouvre une timeline qu'on parcourt à la souris : n'importe quelle image s'affiche immédiatement. En Manim, il faut rejouer la scène depuis le début.

## Qu'est-ce qui a été plus dur ?

- **Tout doit être une fonction du numéro d'image.** Manim est impératif : les updaters gardent un état, et une animation part de là où l'objet se trouve. Ici, rien ne se souvient de l'image précédente, donc il a fallu tout reformuler.
  - Les sauts sont tirés d'avance, avec une graine par candidat ([src/sautiller.ts](src/sautiller.ts)).
  - La position de chaque tête est une fonction par morceaux du temps (`etatTete` dans [src/Primaire.tsx](src/Primaire.tsx)) : sur le corps, en vol, sur la courbe, puis vers le résultat. C'est le morceau le plus délicat du code.
- **Pas de `next_to` ni d'`align_to`.** Manim place les objets selon l'encre des glyphes, alors que le CSS raisonne en boîtes.
  - Les libellés mono se calculent exactement, grâce à la chasse fixe (0,6 em).
  - Les textes en serif s'ancrent par `translate(%)`.
  - Le papier autour d'un mot de largeur inconnue est un SVG étiré à la taille de la boîte (`preserveAspectRatio="none"`).
- **Le vocabulaire d'animation est à réécrire.** `FadeIn(shift=…)`, `lag_ratio`, `Create` et `Write` deviennent des fonctions à la main (`avance`, `echelonne`). `Write` est approché par un fondu lettre à lettre.
- **L'outillage pèse plus lourd dans le projet.** `node_modules` fait 703 Mo, mais le code réellement utile au rendu n'en représente que quelques dizaines :

  | Poste | Taille | Nature |
  |---|---|---|
  | `.remotion/chrome-headless-shell` | 270 Mo | le Chromium sans interface qui rend les images, téléchargé à l'installation |
  | `.cache` | 168 Mo | cache du bundler, supprimable, reconstruit au rendu suivant |
  | `@rspack`, `webpack`, `@babel`, `@esbuild` | ~105 Mo | compilation du TSX en JS pour le navigateur |
  | `@typescript`, `typescript` | ~31 Mo | vérification de types seulement |
  | `@remotion/*` | 61 Mo | dont 29 Mo pour le compositeur natif (qui contient ffmpeg) et 13 Mo pour le studio |
  | `mediabunny` | ~19 Mo | lecture des métadonnées des médias |
  | React, zod, divers | ~50 Mo | |

  Côté Manim, les équivalents existent aussi (Cairo, Pango, LaTeX, ffmpeg), mais ils sont installés au niveau du système, donc invisibles dans le projet.
- **Il faut copier les ressources.** `staticFile()` ne sert que `public/`, donc un script ([ressources.mjs](ressources.mjs)) y copie les ressources avant chaque rendu.
- **Le code est plus long.** Il fait environ 900 lignes, contre 450 pour la scène Manim et une centaine utiles dans `theme.py`. Le JSX est bavard, et il reformule ce que Manim donnait gratuitement.

## Que vaut le rendu obtenu ?

**Visuellement, il est équivalent à la version Manim.**
- La composition est la même à quelques pixels près : même repère, et même échelle des polices à 135/72 px par taille Manim.
- Recadrés à 100 %, le texte et les têtes sont équivalents dans les deux rendus. La trame des coupures est un peu plus douce en Remotion, sans moirage.
- Le mème est lu avec son son, entre 11,28 et 13,58 s d'après `silencedetect`.

**J'ai contrôlé le rendu image par image** (extraction ffmpeg) : planche toutes les 3 s, plus des images en pleine résolution du terminal, du graphe final et de 2017.

**Le contrôle a révélé deux défauts, que j'ai corrigés.** Ils existent aussi dans la version Manim, et ce contrôle est facile à rejouer sur elle.
- **Le bas des corps se décollait du bord.** Au sommet de certains sauts, le bas de Fillon remontait de quelques pixels au-dessus du bord inférieur, parce que le bas de la photo était à −4,3 alors que le saut monte jusqu'à 0,3 et que la bascule soulève un coin. Je l'ai mis à −4,5. Une mesure automatique de la dernière ligne de pixels, sur toutes les images de 1,7 à 10,9 s, trouve maintenant au moins 150 px de corps sur le bord pour chacun.
- **Les points de 2017 sortent de la bande de ± 6 points.** Plusieurs moyennes de Macron sont à plus de 6 points sous son résultat. J'avais d'abord dessiné la bande en aplat, mais les points la débordaient et cela ressemblait à un bug. J'ai donc gardé le choix de la version Manim : ligne de résultat et échelle annoncée, sans aplat. Il faudra peut-être reformuler « chaque bande : 6 points de part et d'autre du résultat ».

**Une seule différence est volontaire.** Sur les courbes de la primaire, chaque point apparaît quand la pointe l'atteint, au lieu d'un fondu échelonné qui le fait apparaître avant le trait.

**Deux choses ne sont pas faites.**
- **Le thème sombre** : la constante `THEME` de [src/charte.ts](src/charte.ts) le permet, mais je ne l'ai ni branché sur une option ni rendu.
- **Le calage sur la voix off**, prévu plus tard dans les deux versions.

## La pile vaut-elle d'être adoptée pour d'autres séquences ?

**Oui pour les séquences faites surtout de mise en page, de texte et de médias** :
- la 0 (accroche) ;
- la 5 (explications des instituts et des chercheurs en colonnes) ;
- la 6 (citations de Bourdieu et Dézé) ;
- le générique ou les cartons.

Le gain y est net : typographie fidèle à la charte sans contournement, médias avec leur son, rendu cinq à six fois plus rapide et aperçu instantané.

**Le calage sur la voix off plaide aussi pour Remotion.** Un `<Audio>` par réplique, et `calculateMetadata` peut fixer la durée des plans d'après les fichiers enregistrés, au lieu d'ajuster des `self.wait()` à la main.

**Les séquences 1 à 4 ne seraient pas dures à porter.** Ce sont presque uniquement des graphes. Relues de près, leurs mécanismes se transposent sans difficulté, et souvent plus simplement.
- **Les `ValueTracker`** (zoom de l'axe en 3, facteur de l'entonnoir en 4) sont déjà des valeurs qui varient dans le temps. En Remotion, `bornes = interpoler(t)` suffit, et le graphe se redessine à chaque image. Il n'y a plus besoin d'updater comme `grille_a_jour`, ni de `clear_updaters()`.
- **Les updaters de couleur** (les points de l'entonnoir en 4, les points qui suivent l'axe en 3) ne dépendent que de la valeur animée. Ils deviennent une simple fonction de celle-ci, sans état.
- **Les tirages aléatoires** (séquence 1) sont le seul vrai point d'attention. Ils viennent de `numpy` avec une graine, et un générateur JS ne donnerait pas les mêmes, alors que la bande à 95 % et l'histogramme doivent refléter les vrais tirages. On les exporterait en JSON par un petit script Python, comme pour la séquence 0 ([exporter_donnees.py](exporter_donnees.py)), ce qui respecte la règle « aucun chiffre en dur ».
- **Les formules** : il n'y a que 4 `MathTex`, toutes dans la séquence 1. KaTeX les affiche sans difficulté. Il serait même plus simple qu'avec LaTeX d'harmoniser les formules avec Newsreader, comme le prévoit le PLAN, parce que le texte autour de la formule est déjà en CSS.
- **Les axes et les graduations** : `d3-scale` donnerait les échelles et les graduations toutes faites.

**Le vrai coût est la relecture, pas la difficulté technique.** Il s'agit de réécrire environ 1 750 lignes déjà validées, puis de refaire la relecture séquence par séquence. C'est un investissement de temps. Il se justifie si l'on veut une seule pile pour le montage et le calage sur la voix off.

Sans migration, les deux piles se mélangent sans difficulté au montage : ce sont des MP4 1080p60 concaténés avec ffmpeg.

**Ma recommandation** :
- passer la 0 en Remotion ;
- écrire la 6, encore à reprendre, directement en Remotion ;
- pour les séquences 1 à 4, décider selon le calage sur la voix off. Si l'on veut caler toute la vidéo sur la voix dans Remotion, les porter est faisable sans difficulté ; sinon, les garder en Manim ne coûte rien.

Le coût d'entrée est déjà payé : charte, papier, terminal, têtes et repère sont dans [src/](src/).
