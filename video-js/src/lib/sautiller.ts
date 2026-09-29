// Sautillements des candidats : le programme des sauts est tiré d'avance (graine par candidat), puis lu à chaque image.
import {hasard} from './outils';

type Saut = {debut: number; duree: number; hauteur: number; dx: number; bascule: number; x0: number};

/**
 * Sauts sans arrêt de `debut` à `fin` (s), à un rythme propre à la graine : hauteurs, durées et pauses tirées au
 * hasard, petits pas de côté rappelés vers la place de départ, bascule en l'air. Le dernier saut se termine avant `fin`.
 */
export const programmeSauts = (graine: number, debut: number, fin: number, basculeMax: number) => {
	const r = hasard(graine);
	const sauts: Saut[] = [];
	let t = debut + r.uniforme(0, 0.3);
	let x = 0;
	for (;;) {
		const dx = Math.min(0.18, Math.max(-0.18, r.normale(0, 0.1) - 0.5 * x));
		const saut = {debut: t, duree: r.uniforme(0.25, 0.5), hauteur: r.uniforme(0.08, 0.3), dx, bascule: r.uniforme(-basculeMax, basculeMax), x0: x};
		if (saut.debut + saut.duree > fin) break;
		sauts.push(saut);
		x += dx;
		t = saut.debut + saut.duree + r.uniforme(0, 0.15);
	}
	return {sauts, xFinal: x};
};

/** Décalage (unités) et angle (radians, sens trigonométrique) du groupe au temps t. */
export const etatSauts = ({sauts, xFinal}: ReturnType<typeof programmeSauts>, t: number) => {
	const i = sauts.findIndex((s) => t < s.debut + s.duree);
	if (i === -1) return {dx: xFinal, dy: 0, angle: 0};
	const s = sauts[i];
	if (t < s.debut) return {dx: s.x0, dy: 0, angle: 0};
	const u = (t - s.debut) / s.duree;
	return {dx: s.x0 + s.dx * u, dy: s.hauteur * Math.sin(Math.PI * u), angle: s.bascule * Math.sin(Math.PI * u)};
};
