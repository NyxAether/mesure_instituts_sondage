// Minutage (s), repris plan par plan de video/scenes/s4_equivalente.py (run_time et wait successifs), regroupé
// par réplique de video/script.md pour un futur calage sur la voix. Les durées d'animation sont en commentaire.

// « Reprenons notre entonnoir : seuls 55 % des écarts y tiennent… Élargissons-le jusqu'à 95 % »
export const ENTONNOIR = {
	entete: 0, // 1 s
	source: 1, // 0,6 s
	axes: 1.6, // 0,8 s
	forme: 2.4, // 1,5 s : entonnoir et nuage de points (fondu échelonné)
	compteur: 3.9, // 0,6 s, puis 2 s d'arrêt
	cadre: 6.5, // 0,5 s
	// Trois étapes de 1,8 s (facteur 2, 4, puis le facteur qui contient 95 %), chacune suivie de 1 s d'arrêt.
	etapes: [7.0, 9.8, 12.6],
	dureeEtape: 1.8,
	verdict: 16.4, // 0,8 s, puis 2,5 s d'arrêt
	sortie: 19.7, // 0,8 s
} as const;

// « L'étude fait ce calcul sondage par sondage… taille annoncée, taille équivalente, témoin »
export const ETUDE = {
	titre: 20.5, // 0,6 s
	lignes: [21.1, 22.8, 24.5], // 0,7 s chacune, puis 1 s d'arrêt
	sortie: 27.7, // après 1,5 s d'arrêt ; 0,8 s
} as const;

// « La taille du sondage n'y change presque rien… »
export const LISSAGE = {
	grille: 28.5, // 1 s
	diagonale: 29.5, // 0,8 s
	reel: 30.3, // 1,5 s : bande, courbe, nom
	valeursReel: 31.8, // 0,5 s, puis 2 s d'arrêt
	temoin: 34.3, // 1,5 s
	valeursTemoin: 35.8, // 0,5 s, puis 2,5 s d'arrêt
	sortie: 38.8, // 0,8 s
} as const;

// « Le temps, lui, compte : plus on s'éloigne de l'élection, plus ça baisse »
export const JOURS = {
	grille: 39.6, // 1 s
	boites: 40.6, // 0,45 s par boîte, l'une après l'autre, puis 2,5 s d'arrêt
	dureeBoite: 0.45,
	sortie: 45.8, // 0,8 s
} as const;

// Écran final : « 2 000 sondés, la précision de 222 »
export const CONSTAT = {
	titre: 46.6, // 1 s
	detail: 47.6, // 0,6 s, puis 0,8 s d'arrêt
	temps: 49.0, // 0,6 s
	fin: 52.1, // après 2,5 s d'arrêt
} as const;
