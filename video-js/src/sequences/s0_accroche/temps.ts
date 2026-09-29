// Minutage (s), repris plan par plan de video/scenes/s0_accroche.py (run_time et wait successifs).
export const DUREE_MEME = 2.35; // 141 images à 60 i/s : l'extrait fait 2,352 s

export const T = {
	corps: 0, // 0,9 s
	tetes: 0.9, // 0,8 s, puis les sauts
	sauts: 1.7,
	banderole: 6.2, // 0,7 s
	ecriture: 6.9, // 1,5 s
	meme: 10.9, // fenêtre en 0,3 s, puis l'extrait
	extrait: 11.2,
	retour: 11.2 + DUREE_MEME,
	graphe: 11.2 + DUREE_MEME + 1.5, // 1,8 s
	noms: 16.85, // 0,6 s
	fillon17: 17.45, // 0,6 s
	courbes: 22.05, // 3,5 s
	saut: 27.55, // 1,6 s
	points: 29.15, // 0,8 s
	soir: 29.95, // 0,6 s
	ecarts: 31.55, // 1 s
	sortiePrimaire: 35.05, // 0,8 s
	accident: 35.85, // 0,6 + 1 + 0,5 s
	p2017: 37.95,
	titre: 46.45,
	fin: 50.95,
};
