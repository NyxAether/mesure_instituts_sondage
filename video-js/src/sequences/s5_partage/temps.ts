// Minutage (s), repris plan par plan de video/scenes/s5_partage.py (run_time et wait successifs).
// Chaque plan est un objet : le début de chaque étape, sa durée en commentaire.

export const TETE = {debut: 0, duree: 1, source: 1}; // en-tête 1 s, puis la source 0,6 s

// L'exemple, si seul le hasard jouait : 1,6 → 11,8 s
export const HASARD = {
	axes: 1.6, // grille, axes, libellés, 0,8 s
	resultats: 2.4, // lignes de résultat et noms des partis, 1 s, puis 1 s d'attente
	mode: 4.4, // « si seul le hasard jouait », 0,5 s
	points: 4.9, // sondages simulés (lag_ratio 0,05) et légende, 2 s, puis 1 s d'attente
	moyennes: 7.9, // moyennes des tirages, 0,8 s
	compense: 8.7, // phrase, 0,6 s, puis 2,5 s d'attente
	fin: 11.8,
};

// Les vrais sondages : tous du même côté : 11,8 → 21,6 s
export const REEL = {
	bascule: 11.8, // moyennes et phrase s'effacent, le libellé change, 0,6 s
	points: 12.4, // les points passent des tirages aux vrais sondages, 2,5 s, puis 1,5 s d'attente
	moyennes: 16.4, // moyennes et écarts au résultat, 0,8 s
	memeCote: 17.2, // phrase, 0,6 s, puis 3 s d'attente
	sortie: 20.8, // fondu de sortie, 0,8 s
	fin: 21.6,
};

// Élection par élection : le consensus d'erreur : 21,6 → 34,0 s
export const CONSENSUS = {
	axe: 21.6, // source, axe, libellés, titre, 1 s
	bande: 22.6, // bande du hasard, 0,8 s, puis 1,5 s d'attente
	points: 24.9, // vrais sondages : médiane et points (lag_ratio 0,01), 2 s, puis 1 s d'attente
	anormaux: 27.9, // points anormaux colorés et compteur, 1,5 s, puis 1,5 s d'attente
	anneau: 30.9, // Royaume-Uni 2015 cerclé, 0,6 s, puis 2 s d'attente
	anneauSortie: 33.5, // 0,5 s
	fin: 34.0,
};

// Le mimétisme : des sondages trop semblables ? : 34,0 → 42,8 s
export const MIMETISME = {
	axe: 34.0, // axe, libellés, titre, repère, 1 s
	points: 35.0, // points (lag_ratio 0,01), 1,5 s, puis 1 s d'attente
	anormaux: 37.5, // points anormaux colorés et grossis, compteur, 1,5 s, puis 3 s d'attente
	sortie: 42.0, // fondu de sortie, 0,8 s
	fin: 42.8,
};

// D'où vient-elle ? : 42,8 → 55,4 s
export const ORIGINE = {
	question: 42.8, // question et prudence, 0,8 s
	instituts: 43.6, // colonne : intitulé et filet 0,5 s, puis trois lignes de 0,5 s + 0,4 s d'attente, puis 1 s d'attente
	chercheurs: 47.8, // idem
	commun: 52.0, // phrase, 0,6 s, puis 2 s d'attente
	sortie: 54.6, // fondu de sortie, 0,8 s
	fin: 55.4,
};
export const PAS_LIGNE = 0.9; // une ligne de colonne : 0,5 s + 0,4 s d'attente
export const DEBUT_LIGNES = 0.5; // les lignes commencent après l'intitulé de la colonne

// Le constat : 55,4 → 64,0 s
export const CONSTAT = {
	constat: 55.4, // titre, 1 s
	detail: 56.4, // 0,6 s, puis 0,8 s d'attente
	moyenne: 57.8, // 0,6 s, puis 1,5 s d'attente
	sans: 59.9, // « Pas de mimétisme », 1 s
	detailSans: 60.9, // 0,6 s, puis 2,5 s d'attente
	fin: 64.0,
};

export const FIN = CONSTAT.fin;
