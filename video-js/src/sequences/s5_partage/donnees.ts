// Données de la séquence 5 (video-js/donnees/s5.json), gardes de cohérence de la scène Manim et essaim de points.
import donnees from '../../../donnees/s5.json';

export const D = donnees;
export const MIM = donnees.mimetisme;
export const ELECTIONS = donnees.elections;
export type Parti = 'conservateurs' | 'travaillistes';
export const PARTIS: Parti[] = ['conservateurs', 'travaillistes'];
export const SONDAGES = donnees.sondages;
export const RESULTAT = donnees.resultat as Record<Parti, number>;
export const SIMULES = donnees.simules as Record<Parti, number[]>;
export const REELS: Record<Parti, number[]> = {
	conservateurs: SONDAGES.map((s) => s.conservateurs),
	travaillistes: SONDAGES.map((s) => s.travaillistes),
};
export const J_MAX = Math.max(...SONDAGES.map((s) => s.jours_avant));
export const FICHE = ELECTIONS[donnees.exemple.index];

// Les constats affichés ne sont vrais que si les données disent la même chose que la scène Manim.
if (FICHE.pays !== donnees.exemple.pays || FICHE.annee !== donnees.exemple.annee || FICHE.tour !== donnees.exemple.tour) {
	throw new Error("l'élection d'exemple n'est plus à l'index indiqué");
}
if (FICHE.sondages !== SONDAGES.length) throw new Error('le nombre de moyennes quotidiennes ne correspond plus à la page');
if (!(RESULTAT.conservateurs > RESULTAT.travaillistes)) throw new Error('partis de l\'exemple inattendus');
if (ELECTIONS.length !== MIM.nb_elections) throw new Error("le nombre d'élections ne correspond plus à la page");
if (SIMULES.conservateurs.length !== SONDAGES.length || SIMULES.travaillistes.length !== SONDAGES.length) {
	throw new Error('un tirage simulé par sondage attendu');
}

/** Hauteurs d'un essaim à une face : chaque point monte jusqu'à ne plus chevaucher les précédents (en unités). */
export const essaim = (xs: number[], diametre: number) => {
	const ordre = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b] || a - b);
	const places: [number, number][] = [];
	const ys = new Array<number>(xs.length).fill(0);
	for (const i of ordre) {
		let y = 0;
		while (places.some(([px, py]) => (xs[i] - px) ** 2 + (y - py) ** 2 < diametre ** 2)) y += diametre / 4;
		ys[i] = y;
		places.push([xs[i], y]);
	}
	return ys;
};
