// Repère, minutage, hasard reproductible, format des nombres et texte positionné.
import React from 'react';
import {EASE, MONO} from './charte';

// Même repère que la version Manim : cadre de 8 unités de haut, origine au centre, y vers le haut.
export const UNITE = 135; // px par unité (1 080 / 8)
export const X = (u: number) => 960 + u * UNITE;
export const Y = (v: number) => 540 - v * UNITE;
// Taille de police Manim en px : Manim compte 72 points par unité.
export const F = (taille: number) => (taille * UNITE) / 72;
// Largeur d'un texte en JetBrains Mono (chasse fixe de 0,6 em).
export const largeurMono = (texte: string, taille: number) => [...texte].length * 0.6 * F(taille);
export const degres = (radians: number) => (-radians * 180) / Math.PI; // sens trigonométrique → sens CSS

export const FPS = 60;

/** Avancement de 0 à 1 d'une animation qui commence à `debut` et dure `duree` secondes, avec la courbe de la charte. */
export const avance = (t: number, debut: number, duree: number, courbe = EASE) =>
	courbe(Math.min(1, Math.max(0, (t - debut) / duree)));

/** Avancement de l'élément k sur n d'une animation échelonnée (lag_ratio de Manim). */
export const echelonne = (t: number, debut: number, duree: number, n: number, k: number, decalage: number) => {
	const d = duree / (1 + decalage * (n - 1));
	return avance(t, debut + k * decalage * d, d);
};

export const lerp = (a: number, b: number, u: number) => a + (b - a) * u;

/** Générateur pseudo-aléatoire reproductible (mulberry32), avec tirages uniformes et normaux. */
export const hasard = (graine: number) => {
	let s = graine >>> 0;
	const suivant = () => {
		s = (s + 0x6d2b79f5) >>> 0;
		let r = Math.imul(s ^ (s >>> 15), 1 | s);
		r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
		return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
	};
	return {
		uniforme: (a: number, b: number) => a + (b - a) * suivant(),
		normale: (moy: number, ecart: number) =>
			moy + ecart * Math.sqrt(-2 * Math.log(1 - suivant())) * Math.cos(2 * Math.PI * suivant()),
	};
};

// Nombres au format français : virgule décimale, espace fine insécable pour les milliers.
export const NBSP = ' ';
export const fr = (x: number, decimales = 1) =>
	x.toFixed(decimales).replace('.', ',').replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
export const pourCent = (v: number) => `${fr(v, Number.isInteger(v) ? 0 : 1)}${NBSP}%`;
export const signe = (v: number) => (v > 0 ? `+${fr(v)}` : `−${fr(-v)}`);

/** Texte placé en px ; (ax, ay) est le point d'ancrage dans sa boîte : 0 à gauche / en haut, 1 à droite / en bas. */
export const Txt: React.FC<{
	x: number;
	y: number;
	taille: number;
	ax?: number;
	ay?: number;
	police?: string;
	couleur?: string;
	opacite?: number;
	style?: React.CSSProperties;
	children: React.ReactNode;
}> = ({x, y, taille, ax = 0, ay = 0, police = MONO, couleur, opacite = 1, style, children}) => (
	<div
		style={{
			position: 'absolute',
			left: x,
			top: y,
			transform: `translate(${-ax * 100}%, ${-ay * 100}%)`,
			fontFamily: police,
			fontSize: F(taille),
			lineHeight: 1,
			whiteSpace: 'pre',
			color: couleur,
			opacity: opacite,
			...style,
		}}
	>
		{children}
	</div>
);
