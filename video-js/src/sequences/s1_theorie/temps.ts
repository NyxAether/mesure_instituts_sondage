// Minutage (s), repris play par play de video/scenes/s1_theorie.py (run_time et wait successifs).
// Regroupé par réplique de video/script.md (section 1) pour un futur calage sur la voix : chaque groupe commence par son début.
import donnees from '../../../donnees/s1.json';

/** Durées (s) des animations et des pauses. */
export const D = {
	tete: 1,
	population: 2, // fondu échelonné des points (le titre de la population : 1 s)
	titrePopulation: 1,
	legende: 0.6,
	axe: 1,
	verite: 0.8,
	echantillon: 0.5,
	// un des tirages lents : mise en évidence, pause, envol vers l'axe, trait, pause
	lentChoix: 1.4,
	lentPause: 1,
	lentEnvol: 1.4,
	lentTrait: 0.4,
	lentAttente: 0.8,
	compteur: 0.4,
	rapidesAttente: 0.6,
	fusion: 1.5,
	lot: 0.9,
	histoAttente: 1,
	zone: 0.6,
	bande: 1.2,
	bandeAttente: 2,
	populationSortie: 0.6,
	formule: 1.2,
	margeTex: 0.8,
	calcul: 0.8,
	formuleAttente: 1.5,
	quatre: 1.6,
	regle: 0.8,
	quatreAttente: 2,
	sortie: 0.8,
	prevision: 1,
	precision: 0.6,
	exception: 0.6,
	conclusionAttente: 1.2,
	invite: 0.5,
	clignote: 0.25, // un demi-clignotement du curseur : disparaît, puis revient
	finAttente: 0.5,
} as const;

export const DUREE_LENT = D.lentChoix + D.lentPause + D.lentEnvol + D.lentTrait + D.lentAttente;
export const NB_CLIGNOTEMENTS = 3;

let curseur = 0;
/** Réserve `duree` s à la suite de tout ce qui précède et renvoie l'instant de début. */
const enchaine = (duree: number) => {
	const debut = curseur;
	curseur += duree;
	return debut;
};

// Ordre d'évaluation : celui des propriétés, donc celui de la scène.
export const T = {
	// Réplique 1 : « Un sondage, c'est une urne. On y tire au hasard mille personnes parmi des millions, et on compte. »
	tete: enchaine(D.tete),
	population: enchaine(D.population),
	legende: enchaine(D.legende),
	axe: enchaine(D.axe),
	verite: enchaine(D.verite),
	echantillon: enchaine(D.echantillon),
	lents: enchaine(donnees.nb_lents * DUREE_LENT),
	// Réplique 2 : « Répétons le tirage des centaines de fois [...] 95 % des tirages tombent à moins de 3 points de la vérité. »
	compteur: enchaine(D.compteur),
	rapides: enchaine(donnees.durees_rapides.reduce((somme, d) => somme + d, 0) + D.rapidesAttente),
	fusion: enchaine(D.fusion),
	lots: enchaine(donnees.lots.length * D.lot + D.histoAttente),
	zone: enchaine(D.zone),
	bande: enchaine(D.bande + D.bandeAttente),
	// Réplique 3 : « Et elle rétrécit quand l'échantillon grandit : avec quatre fois plus de monde, elle est divisée par deux. »
	populationSortie: enchaine(D.populationSortie),
	formule: enchaine(D.formule),
	margeTex: enchaine(D.margeTex),
	calcul: enchaine(D.calcul + D.formuleAttente),
	quatre: enchaine(D.quatre),
	regle: enchaine(D.regle + D.quatreAttente),
	// Réplique 4 : « Voilà ce que prévoit la théorie. Un sondage de mille personnes, c'est plus ou moins trois points [...] Vérifions. »
	sortie: enchaine(D.sortie),
	prevision: enchaine(D.prevision),
	precision: enchaine(D.precision),
	exception: enchaine(D.exception),
	attenteConclusion: enchaine(D.conclusionAttente),
	invite: enchaine(D.invite),
	clignote: enchaine(NB_CLIGNOTEMENTS * 2 * D.clignote + D.finAttente),
	fin: curseur,
};

/** Début (s) du tirage rapide k (k = 0 : le premier, soit le tirage n° nb_lents + 1). */
export const debutsRapides = donnees.durees_rapides.map((_, k, d) => T.rapides + d.slice(0, k).reduce((s, x) => s + x, 0));
