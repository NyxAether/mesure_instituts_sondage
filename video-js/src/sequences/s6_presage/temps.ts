// Minutage (s), repris plan par plan de video/scenes/s6_presage.py (run_time et wait successifs).
// Chaque plan de la scène Manim est un objet : début, puis le début de chaque étape (durées en commentaire).
const FONDU_SORTIE = 0.8;

export const TETE = {debut: 0, duree: 1}; // 1 s

// Précautions : 1 → 8,7 s
export const PRECAUTIONS = {
	base: 1, // barre « tous » et son libellé, 0,8 s
	france: 1.8, // barre France et son libellé, 0,8 s
	limites: [2.6, 3.5, 4.4], // 0,5 s chacune, puis 0,4 s d'attente
	conclusion: 5.3, // 0,6 s
	sortie: 7.9, // 0,6 s + 2 s d'attente, puis fondu de sortie
	fin: 8.7,
};

// Le hasard, plus petite part de l'erreur : 8,7 → 19,6 s
export const HASARD = {
	axes: 8.7, // axe, libellés, légende du hasard, titre, 0,8 s
	hasard: 9.5, // barres du hasard, 1 s, puis 1,5 s d'attente
	observe: 12.0, // barres observées et légende, 1,2 s
	rapports: 13.2, // rapports « × », 1 s (lag_ratio 0,1), puis 1,5 s d'attente
	reste: 15.7, // phrase de conclusion, 0,6 s, puis 2,5 s d'attente
	sortie: 18.8,
	fin: 19.6,
};

// Une critique ancienne : Bourdieu, Dézé : 19,6 → 30,1 s
export const CRITIQUE = {
	bourdieu: 19.6, // 0,6 s
	questions: [20.2, 21.0, 21.8], // 0,5 s chacune, puis 0,3 s d'attente
	artefact: 22.6, // citation, 0,6 s, puis 1,5 s d'attente
	deze: 24.7, // référence et énumération, 0,8 s, puis 1 s d'attente
	photo: 26.5, // « photographie de l'opinion », 0,8 s, puis 2 s d'attente
	sortie: 29.3,
	fin: 30.1,
};

// Les panels en ligne : 30,1 → 37,9 s
export const PANELS = {
	nom: 30.1, // 0,8 s
	citations: [30.9, 32.7], // 0,6 s chacune, puis 1,2 s d'attente
	limite: 34.5, // 0,6 s, puis 2 s d'attente
	sortie: 37.1,
	fin: 37.9,
};

// Photographie, prédiction, présage : 37.9 → 46.7 s
export const MOTS = {
	photographie: 37.9, // mot, note et citation de Gallard, 0,8 s, puis 1,5 s d'attente
	prediction: 40.2, // 0,8 s
	barre: 41.0, // croix sur « photographie », 0,6 s, puis 1,5 s d'attente
	presage: 43.1, // 0,8 s, puis 2 s d'attente
	sortie: 45.9,
	fin: 46.7,
};

// Fin : 46,7 → 55,6 s
export const FORMULE = {
	premiere: 46.7, // 1 s
	seconde: 47.7, // 1 s, puis 1 s d'attente
	ensemble: 49.7, // 0,8 s, puis 1,5 s d'attente
	lien: 52.0, // 0,6 s, puis 3 s d'attente
	fin: 55.6,
};

export const DUREE_TOTALE = FORMULE.fin;
export {FONDU_SORTIE};
export const DUREES = {court: 0.5, moyen: 0.6, long: 0.8};
