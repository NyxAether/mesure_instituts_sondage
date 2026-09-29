// Repère du graphe de la séquence 3 : les trois `Axes` de Manim ont le même cadre, seules les bornes verticales changent.
import donnees from '../../../donnees/s3.json';
import {echelle, X, Y} from '../../lib/axes';
import {UNITE} from '../../lib/outils';

export const Y_ECART = 14; // points ; les écarts exportés sont bornés à ± 14
export const Y_MOYENNE = 2.6; // haut de l'axe une fois zoomé sur les moyennes
export const GRADUATIONS_N = [300, 1_000, 3_000, 10_000, 30_000, 100_000];
export const N_MIN = 200;
export const N_MAX = 150_000;

// Cadre en unités Manim : 10,6 × 4,4 centré en (0,55 ; −0,95).
export const LARGEUR = 10.6;
export const HAUTEUR = 4.4;
export const GAUCHE = 0.55 - LARGEUR / 2;
export const DROITE = GAUCHE + LARGEUR;
export const BAS = -0.95 - HAUTEUR / 2;
export const HAUT = BAS + HAUTEUR;

// Un point de trait Manim vaut 0,01 unité de cadre.
export const TRAIT = 0.01 * UNITE;

const echelleN = echelle([N_MIN, N_MAX], [GAUCHE, DROITE], true);
/** Abscisse (unités) d'une taille d'échantillon. */
export const xu = (n: number) => echelleN(n);
/** Ordonnée (unités) d'une valeur, pour un axe vertical qui va de `bas` à `haut`. */
export const yu = (v: number, bas: number, haut: number) => BAS + ((v - bas) / (haut - bas)) * HAUTEUR;
export {X, Y};

export type Tranche = {n_min: number; n_max: number; n: number; lignes: number; obs: number; th: number};
export const points = donnees.points;
export const tranches: Tranche[] = donnees.par_taille;
// Les écarts sont exportés en fraction : la vidéo les affiche en points de pourcentage.
export const residu = points.map((p) => 100 * p.residu);
export const taille = points.map((p) => p.n);
export const nMed = tranches.map((t) => t.n);
export const obs = tranches.map((t) => 100 * t.obs);
export const th = tranches.map((t) => 100 * t.th);
export const membres = tranches.map((t) => taille.flatMap((n, j) => (n >= t.n_min && n <= t.n_max ? [j] : [])));

// Gardes de cohérence, reprises de la scène Manim.
if (membres.reduce((s, m) => s + m.length, 0) !== points.length) throw new Error('tranches incomplètes');
membres.forEach((m, k) => {
	const moyenne = m.reduce((s, j) => s + Math.abs(residu[j]), 0) / m.length;
	if (Math.abs(moyenne - obs[k]) > 0.005) throw new Error('moyenne de tranche différente de l’export');
});
if (points.length !== donnees.nb_lignes) throw new Error('nombre de points différent de l’export');
