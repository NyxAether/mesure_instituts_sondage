// Briques locales de la séquence 4 (candidates à être partagées) : fondu avec glissement, trait Manim,
// ligne tiretée qui se trace, hauteurs d'encre, textes et lignes de repère.
import React from 'react';
import {C, G} from '../../lib/charte';
import {F, NBSP, UNITE as U, avance, fr, largeurMono} from '../../lib/outils';

/** Épaisseur Manim (stroke_width) en px : cairo compte 0,01 unité par point. */
export const TRAIT = (largeur: number) => largeur * 0.008 * U; // 0,01 unité par point en théorie, un peu moins à l'œil (rendu Manim 480p)
/** Hauteur d'encre des chiffres et capitales en JetBrains Mono, en fraction du corps. */
export const HAUT_ENCRE = 0.73;
export const LIBELLE = C['text-secondary']; // couleur par défaut d'un libelle() de theme.py
export const DISCRET = C['text-muted'];
export const SERIE = G.series[0];

/** Arrondi à l'entier le plus proche (0,5 vers le haut), comme la page. */
export const entier = (x: number) => Math.floor(x + 0.5);
/** Graduation d'écart : « +5 », « 0 », « −5 ». */
export const signeEntier = (v: number) => (v > 0 ? `+${fr(v, 0)}` : v === 0 ? '0' : `−${fr(-v, 0)}`);
export {NBSP};

/** Centre vertical d'un texte de corps `taille` placé sous un bord à la distance `buff` (unités), au sens de next_to(DOWN). */
export const centreSous = (bord: number, buff: number, taille: number) => bord + buff * U + (HAUT_ENCRE * F(taille)) / 2;
/** Centre vertical d'un texte placé au-dessus d'un bord (next_to(UP)). */
export const centreAuDessus = (bord: number, buff: number, taille: number) => bord - buff * U - (HAUT_ENCRE * F(taille)) / 2 - 0.03 * U; // 0,03 : descendantes (p, g, parenthèses)

/** Opacité d'un élément qui apparaît puis disparaît (durées en s). */
export const apparition = (t: number, debut: number, duree: number, sortie?: number, dureeSortie = 0.8) =>
	avance(t, debut, duree) * (sortie === undefined ? 1 : 1 - avance(t, sortie, dureeSortie));

/**
 * FadeIn de Manim : fondu, avec un glissement de `glisse` unités (positif : vers le haut, l'élément part d'en dessous).
 * Les enfants sont placés en coordonnées d'écran.
 */
export const Fondu: React.FC<{opacite: number; glisse?: number; avancement?: number; children: React.ReactNode}> = ({
	opacite,
	glisse = 0,
	avancement = 1,
	children,
}) =>
	opacite <= 0 ? null : (
		<div style={{position: 'absolute', inset: 0, opacity: opacite, transform: `translateY(${glisse * U * (1 - avancement)}px)`}}>{children}</div>
	);

/** Ligne tiretée (DashedLine) qui se trace de `a` à `b` selon `progres` (0 à 1), tirets de `tiret` unités. */
export const LigneTiretee: React.FC<{de: [number, number]; vers: [number, number]; tiret: number; progres: number; couleur: string; epaisseur: number}> = ({
	de,
	vers,
	tiret,
	progres,
	couleur,
	epaisseur,
}) => {
	const [x0, y0] = de;
	const [x1, y1] = vers;
	const longueur = Math.hypot(x1 - x0, y1 - y0);
	const pas = tiret * U; // tiret plein puis intervalle de même longueur
	const vu = longueur * progres;
	const tirets: React.ReactNode[] = [];
	for (let debut = 0; debut < vu; debut += 2 * pas) {
		const fin = Math.min(debut + pas, vu);
		tirets.push(
			<line
				key={debut}
				x1={x0 + ((x1 - x0) * debut) / longueur}
				y1={y0 + ((y1 - y0) * debut) / longueur}
				x2={x0 + ((x1 - x0) * fin) / longueur}
				y2={y0 + ((y1 - y0) * fin) / longueur}
				stroke={couleur}
				strokeWidth={epaisseur}
			/>,
		);
	}
	return <>{tirets}</>;
};

export const CHASSE_ESPACE_FINE = 0.36; // largeur (em) de l'espace fine insécable des milliers dans les libellés mono
/** Largeur d'un libellé mono, avec l'espace fine des milliers plus étroite qu'un caractère. */
export const largeurLibelle = (texte: string, taille: number) => {
	const fines = [...texte].filter((c) => c === ' ').length;
	return largeurMono(texte, taille) - fines * (0.6 - CHASSE_ESPACE_FINE) * F(taille);
};

/** Chemin SVG lissé (courbes de Bézier de type Catmull-Rom) : équivalent de set_points_smoothly de Manim. */
export const cheminLisse = (points: [number, number][]) => {
	let d = `M${points[0][0].toFixed(2)},${points[0][1].toFixed(2)}`;
	for (let k = 0; k < points.length - 1; k++) {
		const p0 = points[Math.max(0, k - 1)];
		const p1 = points[k];
		const p2 = points[k + 1];
		const p3 = points[Math.min(points.length - 1, k + 2)];
		const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
		const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
		d += ` C${c1[0].toFixed(2)},${c1[1].toFixed(2)} ${c2[0].toFixed(2)},${c2[1].toFixed(2)} ${p2[0].toFixed(2)},${p2[1].toFixed(2)}`;
	}
	return d;
};

/** Chemin SVG (segments) à travers des points en px. */
export const chemin = (points: [number, number][], ferme = false) =>
	points.map(([x, y], k) => `${k === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ') + (ferme ? ' Z' : '');

/** Hauteur d'encre chiffrée d'un texte en Newsreader, en fraction du corps. */
export const HAUT_SERIF = 0.65;
/** Ligne « source » sous l'en-tête (Entete : tag de 17, écart de 0,18, titre de 52, puis 0,2 unité de marge). */
export const SOURCE_Y = 0.55 * U + F(17) + 0.18 * U + F(52) + 0.17 * U;
/** Décalage vertical qui ramène le centre d'une boîte Newsreader sur le centre de ses chiffres (ay = 0,5). */
export const CHIFFRES_SERIF = 0.11;
