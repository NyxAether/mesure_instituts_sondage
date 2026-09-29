// Taille équivalente selon la taille réelle : médiane et quartiles des sondages de taille voisine, réels et témoin.
import React from 'react';
import donnees from '../../../donnees/s4.json';
import {echelle} from '../../lib/axes';
import {C, G, SERIF} from '../../lib/charte';
import {F, Txt, UNITE as U, X, Y, avance, fr} from '../../lib/outils';
import {DISCRET, Fondu, HAUT_SERIF, LIBELLE, LigneTiretee, SERIE, SOURCE_Y, TRAIT, CHIFFRES_SERIF, cheminLisse, largeurLibelle, apparition, centreAuDessus, centreSous, chemin, entier} from './commun';
import {LISSAGE as L} from './temps';

const lissage = donnees.lissage;
const lignes = lissage.lignes;
const nMin = lignes[0].n;
const nMax = lignes[lignes.length - 1].n;
const [BAS, HAUT] = [30, 30_000]; // taille équivalente affichée (échelle log)
const GRADUATIONS_Y = [100, 1_000, 10_000];
const GRADUATIONS_X = [1_000, 2_000, 5_000].filter((v) => nMin <= v && v <= nMax);
const TEMOIN = C['text-secondary'];

const echX = echelle([nMin, nMax], [-4.4, 4.6], true);
const echY = echelle([BAS, HAUT], [-2.7, 1.2], true);
const pt = (x: number, y: number): [number, number] => [X(echX(x)), Y(echY(Math.max(BAS, Math.min(HAUT, y))))];

type Mesure = 'optimal_kl' | 'oneshot';
const serie = (mesure: Mesure, quartile: 'q1' | 'med' | 'q3') => lignes.map((l) => l[`${mesure}_${quartile}` as const]);

const largeurLabY = Math.max(...GRADUATIONS_Y.map((v) => largeurLibelle(fr(v, 0), 14)));
const xLabY = pt(nMin, BAS)[0] - 0.15 * U;
const centreLabX = (() => {
	const bords = GRADUATIONS_X.flatMap((v) => [pt(v, BAS)[0] - largeurLibelle(fr(v, 0), 14) / 2, pt(v, BAS)[0] + largeurLibelle(fr(v, 0), 14) / 2]);
	return (Math.min(...bords) + Math.max(...bords)) / 2;
})();

/** Bande interquartile (aplat) et courbe médiane d'une mesure. */
const bande = (mesure: Mesure) =>
	chemin([...lignes.map((l): [number, number] => pt(l.n, l[`${mesure}_q3` as const])), ...[...lignes].reverse().map((l): [number, number] => pt(l.n, l[`${mesure}_q1` as const]))], true);
const courbe = (mesure: Mesure) => cheminLisse(lignes.map((l): [number, number] => pt(l.n, l[`${mesure}_med` as const])));

const Courbe: React.FC<{mesure: Mesure; couleur: string; t: number; debut: number}> = ({mesure, couleur, t, debut}) => {
	const a = avance(t, debut, 1.5);
	return (
		<>
			<path d={bande(mesure)} fill={couleur} fillOpacity={0.15 * a} />
			<path d={courbe(mesure)} fill="none" stroke={couleur} strokeWidth={TRAIT(4)} pathLength={1} strokeDasharray="1 2" strokeDashoffset={1 - a} strokeLinecap="butt" />
		</>
	);
};

const Valeur: React.FC<{x: number; y: number; ax: number; ay: number; couleur: string; children: number}> = ({x, y, ax, ay, couleur, children}) => (
	<Txt x={x} y={y + CHIFFRES_SERIF * F(24)} ax={ax} ay={ay} taille={24} police={SERIF} couleur={couleur}>
		{fr(entier(children), 0)}
	</Txt>
);

export const Courbes: React.FC<{t: number}> = ({t}) => {
	const premier = lignes[0];
	const dernier = lignes[lignes.length - 1];
	const [xa, xb] = [pt(nMin, BAS)[0], pt(nMax, BAS)[0]];
	const diag: [[number, number], [number, number]] = [pt(nMin, nMin), pt(nMax, nMax)];
	const debutReel = pt(nMin, premier.optimal_kl_med);
	const finReel = pt(nMax, dernier.optimal_kl_med);
	const debutTemoin = pt(nMin, premier.oneshot_med);
	const finTemoin = pt(nMax, dernier.oneshot_med);
	const ink = HAUT_SERIF * F(24);
	const aGrille = apparition(t, L.grille, 1);
	return (
		<div style={{position: 'absolute', inset: 0, opacity: 1 - avance(t, L.sortie, 0.8)}}>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				<g opacity={aGrille}>
					{GRADUATIONS_Y.map((v) => (
						<line key={v} x1={xa} x2={xb} y1={pt(nMin, v)[1]} y2={pt(nMin, v)[1]} stroke={G.grid} strokeWidth={TRAIT(1)} />
					))}
					{GRADUATIONS_X.map((v) => (
						<line key={v} x1={pt(v, BAS)[0]} x2={pt(v, BAS)[0]} y1={pt(v, BAS)[1]} y2={pt(v, HAUT)[1]} stroke={G.grid} strokeWidth={TRAIT(1)} />
					))}
					<line x1={xa} x2={xb} y1={pt(nMin, BAS)[1]} y2={pt(nMin, BAS)[1]} stroke={G.axis} strokeWidth={TRAIT(1.5)} />
				</g>
				<LigneTiretee de={diag[0]} vers={diag[1]} tiret={0.08} progres={avance(t, L.diagonale, 0.8)} couleur={DISCRET} epaisseur={TRAIT(1.5)} />
				<Courbe mesure="optimal_kl" couleur={SERIE} t={t} debut={L.reel} />
				<Courbe mesure="oneshot" couleur={TEMOIN} t={t} debut={L.temoin} />
			</svg>
			<Fondu opacite={apparition(t, L.grille, 1)}>
				<Txt x={0.55 * U} y={SOURCE_Y} taille={14} couleur={DISCRET}>
					{`${fr(lissage.effectif, 0)} sondages de la dernière semaine avant le vote · médiane et quartiles, sondages de taille voisine`}
				</Txt>
				{GRADUATIONS_Y.map((v) => (
					<Txt key={v} x={xLabY} y={pt(nMin, v)[1]} ax={1} ay={0.5} taille={14} couleur={LIBELLE}>
						{fr(v, 0)}
					</Txt>
				))}
				<Txt x={xLabY - largeurLabY} y={centreAuDessus(pt(nMin, HAUT)[1], 0.2, 14)} ay={0.5} taille={14} couleur={DISCRET}>
					taille équivalente (échelle log)
				</Txt>
				{GRADUATIONS_X.map((v) => (
					<Txt key={v} x={pt(v, BAS)[0]} y={centreSous(pt(v, BAS)[1], 0.18, 14)} ax={0.5} ay={0.5} taille={14} couleur={LIBELLE}>
						{fr(v, 0)}
					</Txt>
				))}
				<Txt x={centreLabX} y={centreSous(pt(nMin, BAS)[1], 0.18, 14) + 0.15 * U + 0.73 * F(14) + 0.035 * U} ax={0.5} ay={0.5} taille={14} couleur={DISCRET}>
					taille réelle de l’échantillon (échelle log)
				</Txt>
			</Fondu>
			<Fondu opacite={apparition(t, L.diagonale, 0.8)}>
				<Txt x={xb - 0.12 * U} y={pt(nMin, BAS)[1] - 0.12 * U} ax={1} ay={1} taille={13} couleur={DISCRET}>
					pointillés : taille équivalente = taille réelle
				</Txt>
			</Fondu>
			<Fondu opacite={apparition(t, L.reel, 1.5)}>
				<Txt x={pt(1_100, Math.min(...serie('optimal_kl', 'q1')))[0]} y={centreSous(pt(1_100, Math.min(...serie('optimal_kl', 'q1')))[1], 0.2, 15)} ax={0.5} ay={0.5} taille={15} couleur={SERIE}>
					sondages réels
				</Txt>
			</Fondu>
			<Fondu opacite={apparition(t, L.valeursReel, 0.5)}>
				<Valeur x={debutReel[0] + 0.08 * U} y={debutReel[1] - 0.08 * U - ink / 2} ax={0} ay={0.5} couleur={SERIE}>
					{premier.optimal_kl_med}
				</Valeur>
				<Valeur x={finReel[0] + 0.12 * U} y={finReel[1]} ax={0} ay={0.5} couleur={SERIE}>
					{dernier.optimal_kl_med}
				</Valeur>
			</Fondu>
			<Fondu opacite={apparition(t, L.temoin, 1.5)}>
				<Txt x={pt(1_300, Math.max(...serie('oneshot', 'q3')))[0]} y={centreAuDessus(pt(1_300, Math.max(...serie('oneshot', 'q3')))[1], 0.05, 15)} ax={0.5} ay={0.5} taille={15} couleur={TEMOIN}>
					témoin · vrai tirage aléatoire de même taille
				</Txt>
			</Fondu>
			<Fondu opacite={apparition(t, L.valeursTemoin, 0.5)}>
				<Valeur x={debutTemoin[0] + 0.08 * U} y={debutTemoin[1] + 0.08 * U + ink / 2} ax={0} ay={0.5} couleur={TEMOIN}>
					{premier.oneshot_med}
				</Valeur>
				<Valeur x={finTemoin[0] + 0.12 * U} y={finTemoin[1]} ax={0} ay={0.5} couleur={TEMOIN}>
					{dernier.oneshot_med}
				</Valeur>
			</Fondu>
		</div>
	);
};
