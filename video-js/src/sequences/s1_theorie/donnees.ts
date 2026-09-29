// Données de la séquence 1 (export/exporter_s1.py) et calculs qui en découlent : histogrammes, unité de hauteur, marges.
import brut from '../../../donnees/s1.json';
import {NBSP, fr} from '../../lib/outils';

export const D1 = brut;
export const {bornes, hauteur: HAUTEUR, unite_max: UNITE_MAX} = brut;

/** Effectifs par tranche (mêmes règles que np.histogram : [b_k, b_k+1[, la dernière tranche est fermée). */
export const comptes = (valeurs: number[]) => {
	const c = new Array<number>(bornes.length - 1).fill(0);
	for (const v of valeurs) {
		const k = bornes.findIndex((b, i) => i < bornes.length - 1 && v >= b && (v < bornes[i + 1] || (i === bornes.length - 2 && v === bornes[i + 1])));
		if (k < 0) throw new Error(`tirage hors des tranches : ${v}`);
		c[k]++;
	}
	return c;
};
export const tranche = (v: number) => comptes([v]).findIndex((c) => c === 1);
export const centre = (k: number) => (bornes[k] + bornes[k + 1]) / 2;

/** Hauteur d'un tirage : fixe au début, puis réduite pour que la plus haute barre tienne. */
export const unite = (valeurs: number[]) => Math.min(UNITE_MAX, HAUTEUR / Math.max(...comptes(valeurs)));

export const {t1000, t4000} = brut;
const egaux = (a: number[], b: number[]) => a.length === b.length && a.every((x, i) => x === b[i]);
// Gardes de cohérence : nos histogrammes doivent être ceux que numpy a calculés.
if (!egaux(comptes(t1000.slice(0, brut.nb_rapides)), brut.controles.histo_40)) throw new Error('histogramme des 40 premiers tirages incohérent');
for (const lot of brut.lots) {
	if (!egaux(comptes(t1000.slice(0, lot)), brut.controles.histo_lots[String(lot) as keyof typeof brut.controles.histo_lots])) throw new Error(`histogramme du lot ${lot} incohérent`);
}
if (!egaux(comptes(t4000), brut.controles.histo_4000)) throw new Error('histogramme des 4 000 tirages incohérent');
if (brut.votes.length !== brut.colonnes * brut.lignes) throw new Error('population incohérente');
if (brut.echantillons.length !== brut.nb_rapides) throw new Error('nombre d\'échantillons incohérent');

/** Marge d'erreur à 95 %, en points de pourcentage (recalculée : doit valoir celle de l'export). */
export const marge = (n: number, p = brut.p) => 100 * brut.z95 * Math.sqrt((p * (1 - p)) / n);
export const M_PETIT = marge(brut.taille_petit);
export const M_GRAND = marge(brut.taille_grand);
if (Math.abs(M_PETIT - brut.marge_petit) > 1e-5 || Math.abs(M_GRAND - brut.marge_grand) > 1e-5) throw new Error('marges incohérentes');

/** Nombre au format LaTeX : virgule sans espace parasite. */
export const texNombre = (x: number, decimales = 1) => fr(x, decimales).replace(',', '{,}').replaceAll(NBSP, String.raw`\,`);
/** Entier avec espace fine LaTeX entre les milliers (1\,000). */
export const texEntier = (n: number) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, String.raw`\,`);
/** Entier au format de la scène : espace ordinaire entre les milliers (« 1 000 »). */
export const entier = (n: number) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
