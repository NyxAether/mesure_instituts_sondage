// Minutage (s), repris plan par plan de video/scenes/s3_taille.py (run_time et wait successifs) ; le plan est celui de script.md, section 3.
export const T = {
	// En-tête
	entete: 0, // 1 s
	source: 1, // 0,6 s
	// « Le nuage de la séquence 2 »
	axes: 1.6, // 0,8 s : grille, titre de l'axe, graduations de taille
	nuage: 2.4, // 1,5 s, puis 1 s d'attente
	// « Oublions le sens de l'écart, ne gardons que sa taille. »
	pliage: 4.9, // 1,8 s : les écarts négatifs se replient, l'axe descend à zéro ; puis 0,8 s
	// « Regroupons ces points par taille d'échantillon, et calculons l'erreur moyenne de chaque groupe. »
	bandes: 7.5, // 0,8 s, puis 0,6 s
	reduction: 8.9, // 2 s : chaque tranche se réduit à son erreur moyenne ; puis 0,6 s
	retrait: 11.5, // 0,5 s : les bandes et le titre de l'axe s'effacent
	zoom: 12.0, // 2,5 s : l'axe vertical se resserre sur les moyennes ; puis 0,5 s
	// « En théorie, elle devrait fondre »
	theorie: 15.0, // 1,8 s
	etiquettesTheorie: 16.8, // 0,8 s, puis 1,5 s
	// « En réalité, elle reste autour de deux points. »
	observe: 19.1, // 1,5 s
	etiquettesObserve: 20.6, // 0,8 s, puis 1,5 s
	// « Pour les petits sondages, l'erreur est deux fois trop grande. Pour les plus gros, quatre à cinq fois. »
	ecart: 22.9, // 1 s
	rapportGauche: 23.9, // 0,7 s, puis 0,8 s
	rapportDroit: 25.4, // 0,7 s, puis 2 s
	// « ce qui dépasse la théorie, un point et quelque, reste le même à toutes les tailles »
	exces: 28.1, // 1,5 s
	etiquetteExces: 29.6, // 0,7 s, puis 2,5 s
	// Écran final
	sortie: 32.8, // 0,8 s
	constat: 33.6, // 1 s
	detail: 34.6, // 0,6 s, puis 0,8 s
	conclusion: 36.0, // 0,6 s, puis 2,5 s
	fin: 39.1,
};
