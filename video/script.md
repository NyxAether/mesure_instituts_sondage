# Script — « Ce que valent vraiment les sondages »

Voix off d'environ 1 200 mots, soit 8 minutes de parole à 150 mots par minute et environ 9 à 10 minutes avec les pauses. Chaque séquence donne le texte à dire, puis les indications de plan entre crochets et la provenance de chaque chiffre.

Sources des chiffres :
- `erreurs.tous.*` : [docs/data/erreurs.js](../docs/data/erreurs.js), périmètre « tous pays », élections depuis 2000 ;
- `externe.primaire` : [Wikipédia, Sondages sur la primaire française de la droite et du centre de 2016](https://fr.wikipedia.org/wiki/Sondages_sur_la_primaire_fran%C3%A7aise_de_la_droite_et_du_centre_de_2016), hors base.

Références citées :
- Pierre Bourdieu, « L'opinion publique n'existe pas », *Les Temps modernes*, n° 318, janvier 1973, p. 1292-1309 (conférence donnée à Arras en janvier 1972 ; repris dans *Questions de sociologie*, Minuit, 1984). Citations vérifiées sur le [texte reproduit par Acrimed](https://www.acrimed.org/IMG/article_PDF/article_a3938.pdf).
- Alexandre Dézé, *10 leçons sur les sondages politiques*, De Boeck Supérieur, 2022 ([notice Decitre](https://www.decitre.fr/livres/10-lecons-sur-les-sondages-politiques-9782807340312.html)). Leçon n° 3, « Une "photographie de l'opinion" ? », p. 37-44 ([Cairn](https://shs.cairn.info/10-lecons-sur-les-sondages-politiques--9782807340312-page-37?lang=fr)) : l'expression vient des instituts, Dézé la met en question. Le reste du contenu est résumé d'après les présentations du livre : à relire avant d'en dire plus que son objet.
- Alexandre Dézé, *Le sondage d'opinion : outil de la démocratie ou manipulation de l'opinion ?*, Les déjeuners de l'Institut Diderot, février 2022 ([PDF](https://www.institutdiderot.fr/wp-content/uploads/2022/06/DEJEUNER-Sondage-dopinion-Pages.pdf)), conférence du 24 février 2022 qui présente le livre ; lue en entier (p. 11-48). Six problèmes : omniprésence (I), erreurs à répétition (II), représentativité des échantillons (III, p. 16-22 : quotas sur quatre variables sans le diplôme, échantillons effectifs réduits, *access panels* en ligne auto-recrutés, rémunérés, « inscription […] sans condition et sans contrôle » : il s'est inscrit sous une fausse identité, « John »), redressements (IV, p. 22-28 : résultats bruts « faux » selon Roland Cayrol, mémoire du vote, « le détail des opérations reste en effet opaque », « pifomètre » selon Pierre Weill), compréhension et formulation des questions (V, p. 28-35 : « la plupart des sondages politiques placent les personnes interrogées en situation de devoir se prononcer sur des questions qu'ils ne se posent pas forcément », p. 30), contrôle insuffisant (VI). Il conclut que les sondages ne sont ni un outil de la démocratie ni un outil de manipulation de l'opinion : leurs effets sur les électeurs « n'ont jamais pu être démontrés » (p. 46).
- Mathieu Gallard (directeur-adjoint d'Ipsos), *Thématique : les sondages*, Le Nouvel Esprit public, n° 233, 20 février 2022 ([transcription](https://www.lenouvelespritpublic.fr/podcasts/303)), lue en entier. Citations : régionales 2021, les sondages « ont clairement surestimé le vote pour le Rassemblement National, en lien avec une sous-estimation de l'abstention » ; « environ 30% de l'échantillon change d'opinion d'un mois à l'autre » (panel Ipsos–CEVIPOF–*Le Monde*) ; primaire 2016, « dans la toute dernière ligne droite, on a vu une montée très claire de François Fillon » ; « on peut être en retard d'une élection, et ne pas se rendre compte d'une évolution » ; « un sondage n'est pas une prédiction » ; les erreurs sont « beaucoup plus mémorables que leurs réussites », « il faut se garder de surinterpréter ces erreurs ». Sur les panels : l'infiltration « n'est pas impossible, mais elle est si décourageante que ce doit être extrêmement rare » (contrôles de cohérence et de durée).

Crédits photos (séquence 0, portraits détourés, passés en trame de journal et recadrés par [outils/decoupe_portraits.py](outils/decoupe_portraits.py) ; à reprendre dans la description de la vidéo) :
- François Fillon : Thomas Bresson, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2016-10-19_16-14-37_fillon-belfort_(cropped).jpg), CC BY 4.0 ;
- Alain Juppé : Etienne Ansotte / European Union, 2016 / EC - Audiovisual Service, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Alain_Jupp%C3%A9-2016_(cropped).jpg), CC BY 4.0 ;
- Nicolas Sarkozy : European People's Party, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Nicolas_Sarkozy_October_2015_(cropped).jpg), CC BY 2.0 ;
- Emmanuel Macron : Arno Mikkor, EU2017EE Estonian Presidency, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Emmanuel_Macron_(3x4_cropped).jpg), CC BY 2.0 ;
- Marine Le Pen : The Russian Presidential Press and Information Office, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Marine_Le_Pen_(2017-03-24)_01_cropped.jpg), CC BY 4.0.

Les chiffres sont arrondis pour l'oral ; les scènes les liront dans les données et non dans ce texte.

---

## 0. Accroche (0:30)

> Novembre 2016, primaire de la droite. Début novembre, les sondages placent François Fillon troisième, autour de 20 %, loin derrière Alain Juppé. Le soir du premier tour, il obtient 44 %. Juppé, 29.
>
> Accident isolé ? Six mois plus tard, au second tour de la présidentielle, pendant les deux dernières semaines, les sondages se trompent tous du même côté du résultat : Macron sous-estimé, Le Pen surestimée. Pas un seul de l'autre côté.
>
> Alors, que valent vraiment les sondages ? Pour le savoir, on les a confrontés aux résultats de plus de cent élections, dans 45 pays.

[Têtes des trois candidats en coupures de journal, qui suivent la pointe de leur courbe. Courbes Fillon / Juppé / Sarkozy sondage par sondage (Harris 7–9 nov. → Ipsos 18 nov., dix sondages), « Fillon troisième, 17 % », puis sauts en pointillé vers le résultat 44,1 / 28,6 / 20,7, les têtes, en tas sur le dernier sondage, s'envolent vers leur résultat. « Accident isolé ? » Puis présidentielle 2017, 2d tour, les têtes de Macron et Le Pen entrent par la gauche devant leur bande : une bande par candidat autour de sa ligne de résultat (± 6 points), 12 moyennes quotidiennes, toutes du même côté. Titre « Que valent vraiment les sondages ? » avec 102 élections, 45 pays, 15 252 sondages.]

Chiffres : `externe.primaire` = [externe/primaire_2016.json](externe/primaire_2016.json), relevé dans le wikitexte de la révision 233412902 (18 février 2026) de la page citée (Fillon de 17 à 22 % jusqu'au 14 nov., 25 % chez OpinionWay le 15, 27 puis 30 % dans les deux derniers ; résultat 44,1 / 28,6 / 20,7) ; `erreurs.tous.mimetisme.elections` France 2017 tour 2 (12 moyennes quotidiennes sur 14 jours, lues dans `mesure_erreurs/polls.p`, consensus 0,996) ; `mimetisme.nb_elections` = 102, `source.pays` = 45.

> Nuance à garder à l'oral : la primaire est un cas particulier (électorat difficile à cerner, remontée de Fillon dans les derniers jours, visible dans les ultimes sondages). Elle sert d'amorce, pas de preuve.

## 1. La théorie (1:30)

> Un sondage, c'est une urne. On y tire au hasard mille personnes parmi des millions, et on compte. Si on recommençait le tirage, on n'obtiendrait pas exactement le même chiffre : c'est le hasard de l'échantillon.
>
> Mais ce hasard est prévisible. Répétons le tirage des centaines de fois : les résultats se rangent en cloche autour de la vraie valeur. Pour mille personnes et un candidat à 50 %, 95 % des tirages tombent à moins de 3 points de la vérité. C'est la fameuse marge d'erreur.
>
> Et elle rétrécit quand l'échantillon grandit : avec quatre fois plus de monde, elle est divisée par deux.
>
> Voilà ce que prévoit la théorie. Un sondage de mille personnes, c'est plus ou moins trois points, et dans un cas sur vingt seulement, l'écart est plus grand. Vérifions.

[Urne de billes de deux couleurs ; tirages successifs, histogramme qui se construit ; bande à 95 % ±3,1 pts. Formule σ = √(p(1−p)/n) en MathTex, puis marge pour n = 4 000 : ±1,5 pt.]

Chiffres : simulation locale ; 1,96·√(0,25/1000) = 3,1 pts ; 1,96·√(0,25/4000) = 1,5 pt.

## 2. L'entonnoir (1:30)

> La base de données qu'on utilise a été rassemblée par deux politologues, Will Jennings et Christopher Wlezien. Elle compile des dizaines de milliers d'intentions de vote, et le résultat réel de chaque élection. On garde les élections depuis 2000 et, pour commencer, uniquement les sondages publiés dans la dernière semaine avant le vote.
>
> Chaque point, c'est un parti dans un sondage. En hauteur, l'écart entre ce que le sondage annonçait et ce que le parti a vraiment obtenu. En largeur, la taille de l'échantillon.
>
> Si seul le hasard du tirage jouait, 95 % des points resteraient dans cet entonnoir, qui se referme vers la droite.
>
> Ce n'est pas le cas. 45 % des écarts sortent de leur marge d'erreur. Pas les 5 % attendus en théorie : 45. Presque un sur deux.

[Axes, puis points qui apparaissent par vagues ; entonnoir théorique ±L95 pour p = 50 % ; les points hors marge s'allument en couleur d'accent ; compteur qui monte jusqu'à 45 % à côté de « attendu : 5 % ».]

Chiffres : `nuage.nb_lignes` = 1 553, `nuage.nb_sondages` = 424, `nuage.nb_pays` = 32, `nuage.part_hors_marge` = 0,446.

## 3. L'excédent ne diminue pas (1:00)

> Oublions le sens de l'écart, ne gardons que sa taille. Puis regroupons ces points par taille d'échantillon, et calculons l'erreur moyenne de chaque groupe.
>
> En théorie, elle devrait fondre : un peu plus d'un point pour mille personnes, moins d'un demi-point au-delà de cinq mille.
>
> En réalité, elle reste autour de deux points. Pour les petits sondages, l'erreur est deux fois trop grande. Pour les plus gros, quatre à cinq fois.
>
> Elle baisse bien un peu, d'un peu plus d'un demi-point : exactement ce que prévoit le hasard. Mais ce qui dépasse la théorie, un point et quelque, reste le même à toutes les tailles. Interroger plus de monde ne réduit que la part due au hasard. Ce n'est donc pas le hasard qui fait l'essentiel de l'erreur.

[Le nuage de la séquence 2 ; les écarts négatifs se replient vers le haut ; 7 tranches de taille, chacune se réduit à son erreur moyenne ; zoom. Deux courbes : attendue (qui descend, 1,0 → 0,4), observée (plate, 2,3 → 1,8) ; l'écart entre les deux se remplit ; étiquettes « × 2,3 » à gauche, « × 4,8 » à droite. Puis les 7 écarts verticaux entre les courbes s'allument : 1,1 à 1,4 point au-dessus de la théorie, à toutes les tailles. Écran final : « L'excédent d'erreur ne diminue pas », 1,1 à 1,4 point au-dessus de la théorie à toutes les tailles ; seule la baisse prévue par le hasard a lieu (−0,6 point contre −0,6).]

Chiffres : `par_taille[].obs` (2,3 → 1,75 pts), `par_taille[].th` (1,0 → 0,36 pt) ; rapports obs/th ≈ 2,3 et 4,8 ; baisses 0,59 et 0,65 pt ; excédent obs − th de 1,06 à 1,39 pt, sans tendance.

> Nuance : l'excédent est une simple différence (observé − théorique), sans hypothèse sur la façon dont les erreurs se combinent. Si on les suppose indépendantes (combinaison en quadrature), la part non aléatoire baisse un peu (2,1 → 1,7 pt). Ne pas dire à l'oral qu'une « erreur non aléatoire » vaut exactement 1,1 à 1,4 point.

## 4. Combien vaut vraiment un sondage ? (2:00)

> Posons la question autrement. Reprenons notre entonnoir : seuls 55 % des écarts y tiennent, là où la théorie en attend 95. Élargissons-le jusqu'à ce qu'il contienne 95 % des sondages.
>
> Or un entonnoir plus large, c'est exactement celui d'un sondage plus petit. Deux fois plus petit : 70 %. Quatre fois : pas encore. Il faut diviser la taille des sondages par onze. Un sondage de deux mille personnes a l'entonnoir d'un tirage au hasard de moins de deux cents. C'est sa taille équivalente.
>
> L'étude fait ce calcul sondage par sondage, avec des tirages simulés. D'abord un contrôle : sur de vrais tirages aléatoires, la méthode retrouve bien leur taille, 1 973 pour 2 000. Sur les vrais sondages, la taille équivalente médiane est de deux cent vingt. Neuf fois moins que ce qu'ils annoncent.
>
> La taille du sondage n'y change presque rien. Dans la dernière semaine, qu'on interroge huit cents ou huit mille personnes, la taille équivalente médiane reste entre deux et trois cents. Pour de vrais tirages aléatoires, elle suit la taille : de 900 à près de 9 000.
>
> Le temps, lui, compte : plus on s'éloigne de l'élection, plus ça baisse. Normal : l'opinion a le temps de bouger. Dans les cinq derniers jours, la taille équivalente médiane est de l'ordre de trois cents. Un mois avant, moins de cent.

[L'entonnoir de la séquence 2 et ses 1 553 écarts ; compteur « écarts dans leur marge » à 55,4 %, « visé : 95 % ». L'entonnoir s'élargit : « entonnoir d'un sondage 2 fois plus petit » (69,9 %), 4 fois, puis 11 fois (95,0 %) ; les points rentrent dans l'entonnoir. Verdict : « un sondage de 2 000 personnes a l'entonnoir d'un tirage de 184 personnes ». Puis la mesure de l'étude : taille annoncée 2 000, taille équivalente 222 (÷ 9), témoin 1 973. Puis graphe continu, dernière semaine, échelles log : taille équivalente selon la taille réelle, médiane et quartiles des sondages de taille voisine ; sondages réels à plat (240 → 231, entre 213 et 338), témoin le long de la diagonale (907 → 8 708). Puis boîtes par jours avant l'élection (fenêtre d'un mois, médianes 300 → 87). Écran final : « 2 000 sondés, la précision de 222 ».]

Chiffres : `nuage` (1 553 lignes, marge propre de chaque ligne) : 55,4 % dans la marge, facteur 10,9 pour en contenir 95 % (quantile 95 % de (écart / marge)²), soit 2 000 / 10,9 = 184 ; `equivalents.nb_proches` = 859 (fenêtre 14 jours), `equivalents.median_reel` = 2 000, `equivalents.medianes.optimal_kl` = 222, `equivalents.medianes.oneshot` = 1 973 ; boîtes `equivalents.boites` facteur jours, fenêtre ≤ 1 mois (300 à 0–5 jours, 222, 159, 116, 94, puis 87 à 26–30 jours)  ; courbes selon la taille réelle : `donnees.glissante_equivalents(7)`, sur `mesure_erreurs/bss.p` (424 sondages de la dernière semaine), médiane et quartiles des sondages à moins de 0,15 décade de chaque taille, au moins 30 sondages par point (tailles 773 à 8 220). Ce lissage remplace la médiane glissante de la page, qui dépend de l'ordre des sondages de même taille, donc des versions de pandas et numpy.

> L'entonnoir élargi (184) et la mesure de l'étude (222) ne calculent pas la même chose : le premier cherche un facteur commun qui fait entrer 95 % des écarts, la seconde une taille par sondage (médiane des tirages, divergence KL), résumée par sa médiane. Ils donnent le même ordre de grandeur ; à l'oral, dire « moins de deux cents » pour l'un et « deux cent vingt » pour l'autre, sans les présenter comme un même chiffre.

## 5. Pourquoi : l'erreur partagée (1:30)

> Pourquoi les sondages font-ils si mal ? Si chacun se trompait au hasard, certains surestimeraient un parti et d'autres le sous-estimeraient. En faisant la moyenne, les erreurs se compenseraient.
>
> Regardons élection par élection. On mesure si les sondages se trompent tous du même côté : c'est le consensus d'erreur. Au hasard, il devrait être anormalement fort dans 5 % des élections. On le trouve dans 80 % d'entre elles.
>
> Les sondages d'une même élection se trompent ensemble, dans le même sens. C'est pour ça que les agréger ne corrige rien : on fait la moyenne d'une même erreur.
>
> On entend souvent parler de mimétisme, ces instituts qui ajusteraient leurs chiffres pour ne pas trop s'écarter des concurrents. On l'a cherché : des sondages anormalement proches les uns des autres. On n'en trouve pas plus que le hasard n'en produit, environ 3 % des élections. Le mimétisme n'explique donc pas, à lui seul, cette erreur commune.
>
> D'où vient-elle alors ? Ces données ne permettent pas de le dire. Côté instituts, Mathieu Gallard, d'Ipsos, avance des explications de circonstance : l'opinion bouge jusqu'au dernier moment, l'abstention est mal anticipée, et la méthode peut avoir une élection de retard. Les chercheurs regardent plutôt la fabrication du sondage : dès 1973, Pierre Bourdieu rappelait que tout le monde n'a pas d'opinion sur tout ; en 2022, Alexandre Dézé montre des sondés qui répondent à des questions qu'ils ne se posent pas, des échantillons à la représentativité douteuse et des redressements opaques. Dans tous les cas, c'est une erreur que le hasard n'explique pas, et que multiplier les sondages ne dilue pas.

[Royaume-Uni 2015, législatives, 14 derniers jours : lignes du résultat (conservateurs 37,8, travaillistes 31,2) ; d'abord des tirages simulés de même taille, de part et d'autre, dont la moyenne tombe sur le résultat ; puis les points glissent vers les vrais sondages, tous du même côté ; moyennes 33,6 (−4,2) et 33,4 (+2,2). Puis essaim des 102 élections sur l'axe du consensus (0 à 1), bande « au hasard » autour de 0,13, les anormales s'allument (80 %, attendu 5 %), Royaume-Uni 2015 repéré. Puis essaim du resserrement (échelle log, 1 = hasard) : 3 anormales (3 %). Puis deux colonnes : ce qu'avancent les instituts, ce que pointent les chercheurs. Écran final : « Les sondages se trompent ensemble » (80 %, la moyenne ne corrige rien), puis « Pas de mimétisme » (3 %, pas plus qu'au hasard).]

Chiffres : `mimetisme.nb_elections` = 102, `mimetisme.part_consensus` = 0,804, `mimetisme.consensus_median` = 0,56 contre `consensus_hasard_median` = 0,13, `mimetisme.part_resserrement` = 0,029 ; exemple lu dans `mesure_erreurs/polls.p` (`donnees.sondages_election`), 14 points qui sont des moyennes quotidiennes de sondages (la base fusionne ceux d'un même jour).

> Venezuela 2013 retiré : dans la base, ses 11 points sont des tailles de 1 000 ou 2 000 à un sondage par jour (`npolls` = 1), qui descendent régulièrement de 59,6 à 42,1 % pour Maduro en dix jours. Cela ressemble à une série lissée ou interpolée plutôt qu'à « deux camps d'instituts » : son resserrement de 16 vient de cette pente. La même lecture figure sur la page (`docs/erreurs.html`, section mimétisme), à revoir.
>
> Sources de l'écran « D'où vient cette erreur commune ? » (voir les références) : colonne instituts d'après Mathieu Gallard (Ipsos, 2022) ; colonne chercheurs d'après Bourdieu (1973, premier postulat, citation vérifiée) et Dézé (conférence de 2022). « Des questions que les sondés ne se posent pas » est de Dézé (p. 30) et non de Bourdieu. Une seule voix d'institut : c'est un témoignage, pas la position de toute la profession ; à dire comme tel à l'oral.

> Les causes citées sont présentées comme des pistes, pas comme des résultats : l'étude mesure l'erreur, pas son origine.

## 6. Une prédiction plus qu'une photographie, un présage plus qu'une prédiction (1:30)

> Quelques précautions d'abord. La base s'arrête en 2017, elle contient peu de sondages français, et ceux d'un même jour sont parfois fusionnés en une moyenne. Les tendances sont solides à l'échelle des 45 pays ; pour un pays pris seul, elles restent indicatives.
>
> Ce que montrent ces chiffres, c'est que la marge d'erreur ne mesure qu'une chose : le hasard du tirage. Et c'est la plus petite part de l'erreur. Le reste ne se corrige pas en élargissant la marge, parce qu'il ne vient pas du hasard. Il vient de la façon dont le sondage est fabriqué.
>
> Cette critique n'est pas nouvelle. En 1973, Pierre Bourdieu publie « L'opinion publique n'existe pas ». Il y pointe trois postulats implicites des sondages : que tout le monde peut avoir une opinion, que toutes les opinions se valent, et qu'il existe un consensus sur les questions qui méritent d'être posées. L'opinion publique des gros titres serait, écrit-il, « un artefact pur et simple ». Cinquante ans plus tard, le politiste Alexandre Dézé ouvre à son tour la boîte noire des sondages politiques : échantillons par quotas, redressements, formulation des questions. Il y interroge aussi la formule favorite des instituts, le sondage comme « photographie de l'opinion », en lui ajoutant un point d'interrogation.
>
> Il y a un dernier angle mort, plus récent : les panels en ligne, ces volontaires recrutés sur Internet et rémunérés pour répondre. Alexandre Dézé s'y est inscrit sous une fausse identité : une inscription, dit-il, « sans condition et sans contrôle ». Mathieu Gallard, d'Ipsos, répond qu'une infiltration n'est « pas impossible », mais « si décourageante que ce doit être extrêmement rare ». Ces données ne permettent ni de trancher ni de détecter une manipulation.
>
> Car c'est l'argument que les instituts opposent à chaque raté : un sondage ne prédit pas l'élection, il photographie l'opinion à un instant donné. Mais le seul moment où l'on peut vérifier cette photographie, c'est le soir de l'élection : elle est donc lue, et jugée, comme une prédiction. Et une prédiction qui se trompe bien plus souvent que sa marge ne l'annonce, le plus souvent dans le même sens pour tous les instituts, c'est moins une prédiction qu'un présage.
>
> Une prédiction plus qu'une photographie, un présage plus qu'une prédiction. Et quand tous les présages disent la même chose, ce n'est pas une garantie : ils peuvent se tromper ensemble.
>
> Les calculs, les données et les graphiques interactifs sont en lien sous la vidéo.

[Barre des 15 252 sondages depuis 2000, part de la France (467, 3 %), puis les trois limites. Puis, par tranche de taille, deux barres : erreur typique attendue au hasard (ce que mesure la marge) et erreur observée, avec le rapport (×2,3 à ×4,8) ; « le reste ne vient pas du hasard ». Puis les trois postulats de Bourdieu en questions et « un artefact pur et simple », Dézé et « une "photographie de l'opinion" ? ». Panels : Dézé (« sans condition et sans contrôle ») face à Gallard (« si décourageante que ce doit être extrêmement rare »), et la limite « cette étude ne permet ni de détecter ni d'exclure une manipulation ». Puis « photographie » (barré, avec Gallard : « un sondage n'est pas une prédiction »), « prédiction », « présage ». Écran final : la formule, « ils peuvent se tromper ensemble », lien sous la vidéo.]

Chiffres : `selection.sondages` = 15 252, `selection.pays` = 45, `france.selection.sondages` = 467, `source.fin` = 2017 ; `par_taille` : erreur typique `obs` et `th` par tranche, rapports de 2,3 à 4,8. Les barres sont comparées côte à côte et non empilées : des écarts types ne s'additionnent pas.

> Points de vigilance :
> - le paragraphe sur les panels ne doit rien affirmer que cette étude montre : elle ne permet ni de détecter ni d'exclure une manipulation. Il se limite à l'échange Dézé / Gallard ; la manipulation des panels sera le sujet d'une autre vidéo ;
> - « photographie de l'opinion » est la formule des instituts ; Dézé la cite pour la questionner (leçon 3, titre avec point d'interrogation). Ne pas la lui attribuer comme thèse .
