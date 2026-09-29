// Repères de la scène (unités Manim, converties par X/Y de outils.tsx) : axe de l'histogramme, grille de la population.
import {echelle, X, Y} from '../../lib/axes';
import {UNITE} from '../../lib/outils';
import {G, C} from '../../lib/charte';
import {D1, bornes} from './donnees';

/** Épaisseur d'un trait : Manim compte `stroke_width` en centièmes d'unité. */
export const trait = (largeurManim: number) => largeurManim * 0.01 * UNITE;

// Axe de l'histogramme : NumberLine centré en (2.95, -2.55), longueur 7, gradué de bornes[0] à bornes[-1].
export const AXE_CENTRE: [number, number] = [2.95, -2.55];
export const AXE_LONGUEUR = 7;
export const ech = echelle([bornes[0], bornes[bornes.length - 1]], [AXE_CENTRE[0] - AXE_LONGUEUR / 2, AXE_CENTRE[0] + AXE_LONGUEUR / 2]);
/** Abscisse en px d'une valeur de l'axe. */
export const xAxe = (v: number) => X(ech(v));
export const yAxe = Y(AXE_CENTRE[1]);
export const HAUT_TRAIT = 0.35; // hauteur d'un tirage sur l'axe (trait ou point d'arrivée)
export const HAUT_PLACE = 0.35; // hauteur du point d'arrivée des personnes tirées

// Population : 36 x 22 points de rayon 0.042, séparés de 0.06, centrés en (-3.75, -0.35).
export const POP_CENTRE: [number, number] = [-3.75, -0.35];
export const POP_RAYON = 0.042;
export const POP_ESPACE = 0.06;
export const POP_PAS = 2 * POP_RAYON + POP_ESPACE;
export const POP_LARGEUR = D1.colonnes * 2 * POP_RAYON + (D1.colonnes - 1) * POP_ESPACE;
export const POP_HAUTEUR = D1.lignes * 2 * POP_RAYON + (D1.lignes - 1) * POP_ESPACE;
export const POP_GAUCHE = POP_CENTRE[0] - POP_LARGEUR / 2;
/** Centre du point j (colonne) i (ligne, 0 en haut), en unités. */
export const pointPop = (i: number, j: number): [number, number] => [
	POP_CENTRE[0] + (j - (D1.colonnes - 1) / 2) * POP_PAS,
	POP_CENTRE[1] + ((D1.lignes - 1) / 2 - i) * POP_PAS,
];

/** Couleur d'un vote (0 : a, 1 : b) et d'un tirage : celle du vote majoritaire (a au-dessus de 50 %, b en dessous). */
export const couleurVote = (vote: number) => G.series[vote];
export const couleurCote = (v: number) => (v > 50 ? G.series[0] : v < 50 ? G.series[1] : C['text-secondary']);
