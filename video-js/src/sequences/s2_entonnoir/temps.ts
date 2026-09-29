// Minutage (s), repris plan par plan de video/scenes/s2_entonnoir.py (run_time et wait successifs).
// Chaque entrée est le début du plan ; la durée de l'animation figure en commentaire.
export const T = {
	// « La base de données … Jennings et Wlezien … »
	entete: 0, // 1 s
	source: 1, // 0,8 s
	axes: 1.8, // 1,2 s
	// « Chaque point, c'est un parti dans un sondage. En hauteur, l'écart … En largeur, la taille de l'échantillon. »
	pointExemple: 3, // 0,6 s
	noteExemple: 3.6, // 0,9 s
	barreExemple: 4.5, // 0,8 s, puis 2,5 s de pause
	retraitExemple: 7.8, // 0,6 s
	// « Si seul le hasard du tirage jouait, 95 % des points resteraient dans cet entonnoir »
	entonnoir: 8.4, // 1,2 s
	cadreTheorie: 9.6, // 0,6 s
	tirages: 10.2, // 2,5 s
	compteur: 12.7, // 0,6 s, puis 2 s de pause
	// « Ce n'est pas le cas. 45 % des écarts sortent de leur marge d'erreur. »
	cadreReel: 15.3, // 0,6 s
	realite: 15.9, // 3 s, puis 2,5 s de pause
	sortieGraphe: 21.4, // 0,8 s
	// « Pas les 5 % attendus en théorie : 45. Presque un sur deux. »
	constat: 22.2, // 1 s
	phrase: 23.2, // 0,6 s, puis 0,8 s de pause
	rappel: 24.6, // 0,6 s, puis 2,5 s de pause
	fin: 27.7,
};
