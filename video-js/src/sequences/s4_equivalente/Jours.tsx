// Taille équivalente médiane selon le nombre de jours avant l'élection : une boîte (quartiles, médiane) par tranche.
import React from 'react';
import donnees from '../../../donnees/s4.json';
import {echelle} from '../../lib/axes';
import {C, G, SERIF} from '../../lib/charte';
import {F, Txt, UNITE as U, X, Y, avance, fr} from '../../lib/outils';
import {DISCRET, Fondu, LIBELLE, SERIE, SOURCE_Y, TRAIT, CHIFFRES_SERIF, largeurLibelle, apparition, centreAuDessus, centreSous, entier} from './commun';
import {JOURS as J} from './temps';

const jours = donnees.jours;
const Y0 = -2.7;
const Y_HAUT = 1.2;
const V_MAX = 700;
const GAUCHE = -4.4;
const DROITE = 5.2;
const LARGEUR = 0.75; // unités
const xs = jours.map((_, k) => -3.5 + (k * (4.1 + 3.5)) / (jours.length - 1));
const echV = echelle([0, V_MAX], [Y0, Y_HAUT]);
const yEcran = (v: number) => Y(echV(v));
const GRADUATIONS = [0, 200, 400, 600]; // range(0, v_max + 1, 200) de la scène Manim
const largeurLabY = Math.max(...GRADUATIONS.map((v) => largeurLibelle(fr(v, 0), 14)));

export const Jours: React.FC<{t: number}> = ({t}) => {
	const aGrille = apparition(t, J.grille, 1);
	const bordsX = jours.flatMap((b, k) => [X(xs[k]) - largeurLibelle(b.tranche, 15) / 2, X(xs[k]) + largeurLibelle(b.tranche, 15) / 2]);
	const yLabX = centreSous(yEcran(0), 0.18, 15);
	return (
		<div style={{position: 'absolute', inset: 0, opacity: 1 - avance(t, J.sortie, 0.8)}}>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0, opacity: aGrille}}>
				{GRADUATIONS.map((v) => (
					<line key={v} x1={X(GAUCHE)} x2={X(DROITE)} y1={yEcran(v)} y2={yEcran(v)} stroke={v === 0 ? G.axis : G.grid} strokeWidth={TRAIT(v === 0 ? 1.5 : 1)} />
				))}
			</svg>
			<Fondu opacite={aGrille}>
				<Txt x={0.55 * U} y={SOURCE_Y} taille={14} couleur={DISCRET}>
					{`${fr(donnees.effectif_30_jours, 0)} sondages du dernier mois avant le vote · taille équivalente médiane et quartiles`}
				</Txt>
				{GRADUATIONS.map((v) => (
					<Txt key={v} x={X(GAUCHE) - 0.15 * U} y={yEcran(v)} ax={1} ay={0.5} taille={14} couleur={LIBELLE}>
						{fr(v, 0)}
					</Txt>
				))}
				<Txt x={X(GAUCHE) - 0.15 * U - largeurLabY} y={centreAuDessus(yEcran(V_MAX), 0.2, 14)} ay={0.5} taille={14} couleur={DISCRET}>
					taille équivalente
				</Txt>
				{jours.map((b, k) => (
					<Txt key={b.tranche} x={X(xs[k])} y={yLabX} ax={0.5} ay={0.5} taille={15} couleur={LIBELLE}>
						{b.tranche}
					</Txt>
				))}
				<Txt x={(Math.min(...bordsX) + Math.max(...bordsX)) / 2} y={yLabX + 0.15 * U + 0.73 * F(15) + 0.035 * U} ax={0.5} ay={0.5} taille={14} couleur={DISCRET}>
					jours avant l’élection
				</Txt>
			</Fondu>
			{jours.map((b, k) => {
				const a = avance(t, J.boites + k * J.dureeBoite, J.dureeBoite);
				const [bas, haut, med] = [yEcran(b.q1), yEcran(b.q3), yEcran(b.med)];
				return (
					<Fondu key={b.tranche} opacite={a} glisse={0.1} avancement={a}>
						<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
							<rect x={X(xs[k]) - (LARGEUR * U) / 2} y={haut} width={LARGEUR * U} height={bas - haut} fill={SERIE} fillOpacity={0.15} stroke={SERIE} strokeWidth={TRAIT(2)} />
							<line x1={X(xs[k]) - (LARGEUR * U) / 2} x2={X(xs[k]) + (LARGEUR * U) / 2} y1={med} y2={med} stroke={SERIE} strokeWidth={TRAIT(5)} />
						</svg>
						<Txt x={X(xs[k]) + (LARGEUR * U) / 2 + 0.12 * U} y={med + CHIFFRES_SERIF * F(30)} ay={0.5} taille={30} police={SERIF} couleur={C['text-primary']}>
							{fr(entier(b.med), 0)}
						</Txt>
					</Fondu>
				);
			})}
		</div>
	);
};
