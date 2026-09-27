# Script — « Ce que valent vraiment les sondages »

Voix off d'environ 1 200 mots, soit 8 minutes de parole à 150 mots par minute et environ 9 à 10 minutes avec les pauses. Chaque séquence donne le texte à dire, puis les indications de plan entre crochets et la provenance de chaque chiffre.

Sources des chiffres :
- `erreurs.tous.*` : [docs/data/erreurs.js](../docs/data/erreurs.js), périmètre « tous pays », élections depuis 2000 ;
- `externe.primaire` : [Wikipédia, Sondages sur la primaire française de la droite et du centre de 2016](https://fr.wikipedia.org/wiki/Sondages_sur_la_primaire_fran%C3%A7aise_de_la_droite_et_du_centre_de_2016), hors base.

Références citées :
- Pierre Bourdieu, « L'opinion publique n'existe pas », *Les Temps modernes*, n° 318, janvier 1973, p. 1292-1309 (conférence donnée à Arras en janvier 1972 ; repris dans *Questions de sociologie*, Minuit, 1984). Citations vérifiées sur le [texte reproduit par Acrimed](https://www.acrimed.org/IMG/article_PDF/article_a3938.pdf).
- Alexandre Dézé, *10 leçons sur les sondages politiques*, De Boeck Supérieur, 2022 ([notice Decitre](https://www.decitre.fr/livres/10-lecons-sur-les-sondages-politiques-9782807340312.html)). Leçon n° 3, « Une "photographie de l'opinion" ? », p. 37-44 ([Cairn](https://shs.cairn.info/10-lecons-sur-les-sondages-politiques--9782807340312-page-37?lang=fr)) : l'expression vient des instituts, Dézé la met en question. Le reste du contenu est résumé d'après les présentations du livre : à relire avant d'en dire plus que son objet.

Les chiffres sont arrondis pour l'oral ; les scènes les liront dans les données et non dans ce texte.

---

## 0. Accroche (0:30)

> Novembre 2016, primaire de la droite. Début novembre, les sondages placent François Fillon troisième, autour de 20 %, loin derrière Alain Juppé. Le soir du premier tour, il obtient 44 %. Juppé, 29.
>
> Accident isolé ? Six mois plus tard, au second tour de la présidentielle, les douze derniers sondages se trompent tous du même côté du résultat. Pas un seul de l'autre côté.
>
> Alors, que valent vraiment les sondages ? Pour le savoir, on les a confrontés aux résultats de plus de cent élections, dans 45 pays.

[Barres Fillon / Juppé / Sarkozy sondage par sondage (Harris 7–9 nov. → Ipsos 18 nov.), puis bascule vers le résultat 44,1 / 28,6 / 20,7. Puis présidentielle 2017, 2d tour : ligne du résultat, 12 points tous du même côté. Titre.]

Chiffres : `externe.primaire` (Fillon 17–22 % jusqu'au 15 nov., résultat 44,1 / 28,6 / 20,7) ; `erreurs.tous.mimetisme.elections` France 2017 tour 2 (12 sondages, consensus 0,996) ; `mimetisme.nb_elections` = 102, `source.pays` = 45.

> Nuance à garder à l'oral : la primaire est un cas particulier (électorat difficile à cerner, remontée de Fillon dans les derniers jours, visible dans les ultimes sondages). Elle sert d'amorce, pas de preuve.

## 1. La promesse (1:30)

> Un sondage, c'est une urne. On y tire au hasard mille personnes parmi des millions, et on compte. Si on recommençait le tirage, on n'obtiendrait pas exactement le même chiffre : c'est le hasard de l'échantillon.
>
> Mais ce hasard est prévisible. Répétons le tirage des centaines de fois : les résultats se rangent en cloche autour de la vraie valeur. Pour mille personnes et un candidat à 50 %, 95 % des tirages tombent à moins de 3 points de la vérité. C'est la fameuse marge d'erreur.
>
> Et elle rétrécit quand l'échantillon grandit : avec quatre fois plus de monde, elle est divisée par deux.
>
> Voilà la promesse. Un sondage de mille personnes, c'est plus ou moins trois points, et dans un cas sur vingt seulement, un peu plus. Vérifions-la.

[Urne de billes de deux couleurs ; tirages successifs, histogramme qui se construit ; bande à 95 % ±3,1 pts. Formule σ = √(p(1−p)/n) en MathTex, puis marge pour n = 4 000 : ±1,5 pt.]

Chiffres : simulation locale ; 1,96·√(0,25/1000) = 3,1 pts ; 1,96·√(0,25/4000) = 1,5 pt.

## 2. L'entonnoir (1:30)

> La base de données qu'on utilise a été rassemblée par deux politologues, Will Jennings et Christopher Wlezien. Elle compile des dizaines de milliers d'intentions de vote, et le résultat réel de chaque élection. On garde les élections depuis 2000 et, pour commencer, uniquement les sondages publiés dans la dernière semaine avant le vote.
>
> Chaque point, c'est un parti dans un sondage. En hauteur, l'écart entre ce que le sondage annonçait et ce que le parti a vraiment obtenu. En largeur, la taille de l'échantillon.
>
> Si la promesse était tenue, 95 % des points resteraient dans cet entonnoir, et il se refermerait vers la droite.
>
> Ce n'est pas le cas. 45 % des écarts sortent de leur marge d'erreur. Pas 5 %. 45. Presque un sur deux.

[Axes, puis points qui apparaissent par vagues ; entonnoir théorique ±L95 pour p = 50 % ; les points hors marge s'allument en couleur d'accent ; compteur qui monte jusqu'à 45 % à côté de « attendu : 5 % ».]

Chiffres : `nuage.nb_lignes` = 1 553, `nuage.nb_sondages` = 424, `nuage.nb_pays` = 32, `nuage.part_hors_marge` = 0,446.

## 3. L'erreur ne baisse pas (1:00)

> Regroupons ces points par taille d'échantillon, et calculons l'erreur moyenne de chaque groupe.
>
> En théorie, elle devrait fondre : un peu plus d'un point pour mille personnes, moins d'un demi-point au-delà de cinq mille.
>
> En réalité, elle reste bloquée autour de deux points, quelle que soit la taille. Pour les petits sondages, l'erreur est deux fois trop grande. Pour les plus gros, quatre à cinq fois.
>
> Interroger plus de monde ne sert presque à rien. Ce n'est donc pas le hasard qui fait l'essentiel de l'erreur.

[Deux courbes : attendue (qui descend), observée (plate) ; l'écart entre les deux se remplit ; étiquettes « ×2 » à gauche, « ×5 » à droite.]

Chiffres : `par_taille[].obs` (2,3 → 1,75 pts), `par_taille[].th` (1,0 → 0,36 pt) ; rapports obs/th ≈ 2,3 et 4,8.

## 4. Combien vaut vraiment un sondage ? (2:00)

> Posons la question autrement. Un sondage réel, de mille ou deux mille personnes, se trompe d'une certaine quantité. Quelle taille faudrait-il à une urne parfaitement aléatoire pour se tromper autant ?
>
> Pour chaque sondage, on simule des tirages au hasard à partir du vrai résultat de l'élection. On fait varier la taille de l'urne, jusqu'à trouver celle dont les tirages sont, en médiane, aussi loin de la vérité que le sondage. C'est sa taille équivalente.
>
> D'abord, un contrôle. On remplace les sondages par de vrais tirages aléatoires, et on applique la même méthode. Elle retrouve bien leur taille : autour de deux mille. La méthode fonctionne.
>
> Maintenant, les vrais sondages. Leur taille médiane est de deux mille personnes. Leur taille équivalente : environ deux cent vingt. Un sondage se comporte comme un tirage au sort de quelques centaines de personnes. Dix fois moins que ce qu'il annonce.
>
> Et plus on s'éloigne de l'élection, plus ça baisse. Normal : l'opinion a le temps de bouger. Dans les cinq derniers jours, la taille équivalente médiane est de l'ordre de trois cents. Un mois avant, moins de cent.

[Urne dont on réduit la taille (2 000 → 1 000 → 500 → 220), à côté la distribution des tirages qui s'élargit jusqu'à atteindre l'écart du sondage réel. Puis deux barres : témoin ≈ 1 973, sondages réels ≈ 222. Puis boîtes par jours avant l'élection.]

Chiffres : `equivalents.nb_sondages` = 15 252 (fenêtre 14 jours pour la médiane), `equivalents.median_reel` = 2 000, `equivalents.medianes.optimal_kl` = 222, `equivalents.medianes.oneshot` = 1 973 ; boîtes `equivalents.boites` facteur jours, fenêtre ≤ 1 mois (≈ 300 à 0–5 jours, 87 à 26–30 jours, d'après le texte de la page — à relire dans les données).

## 5. Pourquoi : l'erreur partagée (1:30)

> Pourquoi les sondages font-ils si mal ? Si chacun se trompait au hasard, certains surestimeraient un parti et d'autres le sous-estimeraient. En faisant la moyenne, les erreurs se compenseraient.
>
> Regardons élection par élection. On mesure si les sondages se trompent tous du même côté : c'est le consensus d'erreur. Au hasard, il devrait être anormalement fort dans 5 % des élections. On le trouve dans 80 % d'entre elles.
>
> Les sondages d'une même élection se trompent ensemble, dans le même sens. C'est pour ça que les agréger ne corrige rien : on fait la moyenne d'une même erreur.
>
> On entend souvent parler de mimétisme, ces instituts qui ajusteraient leurs chiffres pour ne pas trop s'écarter des concurrents. On l'a cherché : des sondages anormalement proches les uns des autres. On n'en trouve pas plus que le hasard n'en produit, environ 3 % des élections. Le mimétisme n'explique donc pas, à lui seul, cette erreur commune.
>
> D'où vient-elle alors ? Ces données ne permettent pas de le dire. Les candidats ne manquent pas : des électeurs difficiles à joindre, des indécis qui tranchent au dernier moment, des redressements calés sur les mêmes élections passées, des questions qui orientent les réponses, ou des panels exposés aux mêmes biais. Tous laissent la même trace : une erreur que le hasard n'explique pas, et que multiplier les sondages ne dilue pas.
>
> Et parfois, c'est l'inverse : au Venezuela en 2013, les sondages s'éparpillent de part et d'autre du résultat, comme deux camps d'instituts qui ne mesuraient pas le même pays.

[Pour une élection (Royaume-Uni 2015 ou France 2017, 2d tour) : ligne du résultat, points de sondages tous du même côté ; flèche de « moyenne » qui tombe elle aussi à côté. Puis nuage consensus × resserrement, 102 points, les anormaux s'allument. Zoom Venezuela 2013.]

Chiffres : `mimetisme.nb_elections` = 102, `mimetisme.part_consensus` = 0,804, `mimetisme.consensus_median` = 0,56 contre `consensus_hasard_median` = 0,13, `mimetisme.part_resserrement` = 0,029 ; Venezuela 2013 : consensus 0,05, resserrement 16.

> Les causes citées sont présentées comme des pistes, pas comme des résultats : l'étude mesure l'erreur, pas son origine.

## 6. Une prédiction plus qu'une photographie, un présage plus qu'une prédiction (1:30)

> Quelques précautions d'abord. La base s'arrête en 2017, elle contient peu de sondages français, et ceux d'un même jour sont parfois fusionnés en une moyenne. Les tendances sont solides à l'échelle des 45 pays ; pour un pays pris seul, elles restent indicatives.
>
> Ce que montrent ces chiffres, c'est que la marge d'erreur ne mesure qu'une chose : le hasard du tirage. Et c'est la plus petite part de l'erreur. Le reste ne se corrige pas en élargissant la marge, parce qu'il ne vient pas du hasard. Il vient de la façon dont le sondage est fabriqué.
>
> Cette critique n'est pas nouvelle. En 1973, Pierre Bourdieu publie « L'opinion publique n'existe pas ». Il y pointe trois postulats implicites des sondages : que tout le monde peut avoir une opinion, que toutes les opinions se valent, et qu'il existe un consensus sur les questions qui méritent d'être posées. L'opinion publique des gros titres serait, écrit-il, « un artefact pur et simple ». Cinquante ans plus tard, le politiste Alexandre Dézé ouvre à son tour la boîte noire des sondages politiques : échantillons par quotas, redressements, formulation des questions. Il y interroge aussi la formule favorite des instituts, le sondage comme « photographie de l'opinion », en lui ajoutant un point d'interrogation.
>
> Il y a un dernier angle mort, plus récent : les panels en ligne. Si une erreur peut être partagée par tous les instituts, elle peut aussi être provoquée. [À COMPLÉTER : risques d'influence externe et de manipulation des panels, d'après tes travaux.]
>
> Car c'est l'argument que les instituts opposent à chaque raté : un sondage ne prédit pas l'élection, il photographie l'opinion à un instant donné. Mais le seul moment où l'on peut vérifier cette photographie, c'est le soir de l'élection : elle est donc lue, et jugée, comme une prédiction. Et une prédiction qui se trompe bien plus souvent que sa marge ne l'annonce, le plus souvent dans le même sens pour tous les instituts, c'est moins une prédiction qu'un présage.
>
> Une prédiction plus qu'une photographie, un présage plus qu'une prédiction. Et quand tous les présages disent la même chose, ce n'est pas une garantie : ils peuvent se tromper ensemble.
>
> Les calculs, les données et les graphiques interactifs sont en lien sous la vidéo.

[Liste ou carte des 45 pays, bandeau « France : peu de données ». Puis décomposition visuelle de l'erreur : une petite part « hasard du tirage » (la marge annoncée) et une grande part « fabrication du sondage », sans chiffrer cette dernière. Couverture et citation de Bourdieu (1973), couverture du livre de Dézé (2022). Plan sur les panels (à définir avec tes travaux). Lien vers la page.]

Chiffres : `source.pays` = 45, `source.fin` = 2017 ; rapports obs/th de `par_taille` (×2 à ×5) pour la décomposition hasard / reste.

> Points de vigilance :
> - le paragraphe sur les panels ne doit rien affirmer que cette étude montre : elle ne permet ni de détecter ni d'exclure une manipulation. Il s'appuie sur tes travaux, à citer à l'écran ;
> - « photographie de l'opinion » est la formule des instituts ; Dézé la cite pour la questionner (leçon 3, titre avec point d'interrogation). Ne pas la lui attribuer comme thèse .
