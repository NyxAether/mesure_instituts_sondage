// « Accident isolé ? », présidentielle 2017 (second tour), puis le titre.
import React from 'react';
import {useCurrentFrame} from 'remotion';
import donnees from '../../../donnees/s0.json';
import {C, SERIF, couleurs} from '../../lib/charte';
import {Tete} from './Primaire';
import {T} from './temps';
import {F, FPS, NBSP, Txt, UNITE as U, X, Y, avance, echelonne, fr, lerp, pourCent} from '../../lib/outils';

const p2017 = donnees.presidentielle_2017;
type Cand = 'macron' | 'lepen';
const CANDS: Cand[] = ['macron', 'lepen'];
const NOMS: Record<Cand, string> = {macron: 'Macron', lepen: 'Le Pen'};
// Deux bandes, une par candidat, chacune centrée sur le résultat (± 6 points) : (bas, haut) en unités.
const BANDES: Record<Cand, [number, number]> = {macron: [0.9, 2.6], lepen: [-2.4, -0.7]};
const DEMI_BANDE = 6;
const INCLINAISONS: Record<Cand, number> = {macron: 0.07, lepen: -0.08};
const J_MAX = p2017.jours_max;

const pt = (cand: Cand, jours: number, v: number): [number, number] => {
	const [b, h] = BANDES[cand];
	const r = p2017.resultat[cand];
	return [-5.2 + ((J_MAX - jours) / (J_MAX - 1)) * 8.4, b + ((v - (r - DEMI_BANDE)) / (2 * DEMI_BANDE)) * (h - b)];
};

// Le constat affiché n'est vrai que si les données le disent.
const tousDuMemeCote =
	p2017.sondages.every((s) => s.macron < p2017.resultat.macron) && p2017.sondages.every((s) => s.lepen > p2017.resultat.lepen);
if (!tousDuMemeCote) throw new Error("l'exemple 2017 n'est plus tout d'un côté");

const Accident: React.FC<{t: number}> = ({t}) => (
	<Txt x={960} y={540} ax={0.5} ay={0.5} taille={40} police={SERIF} couleur={C['text-primary']} opacite={avance(t, T.accident, 0.6) - avance(t, T.accident + 1.6, 0.5)}>
		{`Accident isolé${NBSP}?`}
	</Txt>
);

const Presidentielle: React.FC<{t: number}> = ({t}) => {
	const S = T.p2017;
	const elements = avance(t, S, 0.8) * (1 - avance(t, S + 7.7, 0.8));
	const sortie = 1 - avance(t, S + 7.7, 0.8);
	// Chaque tête entre par la gauche et se pose devant sa bande, puis ses points apparaissent.
	const entrees: Record<Cand, number> = {macron: S + 1.2, lepen: S + 2.9};
	const jours = [J_MAX, 10, 5, 2];
	const hautJours = Y(pt('lepen', 0, p2017.resultat.lepen - DEMI_BANDE)[1]) + 0.1 * U;
	const [xj0, xj1] = [X(pt('lepen', J_MAX, 0)[0]), X(pt('lepen', 2, 0)[0])];
	return (
		<div style={{position: 'absolute', inset: 0}}>
			<div style={{position: 'absolute', inset: 0, opacity: elements}}>
				<Txt x={0.55 * U} y={0.55 * U} taille={16} couleur={C['text-muted']}>
					{`présidentielle 2017 · 2d tour · ${J_MAX} derniers jours, une moyenne des sondages par jour`}
				</Txt>
				<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
					{CANDS.map((cand) => {
						const r = p2017.resultat[cand];
						const [x0, y0] = pt(cand, J_MAX + 0.5, r + DEMI_BANDE);
						const [x1, y1] = pt(cand, 0.5, r - DEMI_BANDE);
						return (
							<g key={cand}>
								<line x1={X(x0)} x2={X(x1)} y1={Y(pt(cand, 0, r)[1])} y2={Y(pt(cand, 0, r)[1])} stroke={couleurs[cand]} strokeWidth={2.5} strokeDasharray="13.5 13.5" />
							</g>
						);
					})}
				</svg>
				{CANDS.map((cand) => {
					const r = p2017.resultat[cand];
					const [x, y] = pt(cand, 0.5, r);
					return (
						<div key={cand} style={{position: 'absolute', left: X(x) + 0.2 * U, top: Y(y), transform: 'translateY(-50%)', color: couleurs[cand]}}>
							<Txt x={0} y={0} taille={15} style={{position: 'relative'}}>
								{NOMS[cand].toLowerCase()}
							</Txt>
							<Txt x={0} y={0} taille={13} style={{position: 'relative', marginTop: 0.05 * U}}>
								{`résultat ${pourCent(Math.round(r * 10) / 10)}`}
							</Txt>
						</div>
					);
				})}
				{jours.map((j) => (
					<Txt key={j} x={X(pt('lepen', j, 0)[0])} y={hautJours} ax={0.5} taille={12} couleur={C['text-secondary']}>
						{fr(j, 0)}
					</Txt>
				))}
				<Txt x={(xj0 + xj1) / 2} y={hautJours + F(12) + 0.08 * U} ax={0.5} taille={12} couleur={C['text-muted']}>
					jours avant l’élection
				</Txt>
			</div>
			<Txt x={0.55 * U} y={0.55 * U + F(16) + 0.15 * U} taille={12} couleur={C['text-muted']} opacite={avance(t, S + 0.8, 0.4) * sortie}>
				{`chaque bande${NBSP}: ${DEMI_BANDE} points de part et d’autre du résultat`}
			</Txt>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0, opacity: sortie}}>
				{CANDS.map((cand) =>
					p2017.sondages.map((s, k) => {
						const [x, y] = pt(cand, s.jours_avant, s[cand]);
						return (
							<circle
								key={`${cand}${k}`}
								cx={X(x)}
								cy={Y(y)}
								r={0.07 * U}
								fill={couleurs[cand]}
								opacity={echelonne(t, entrees[cand] + 0.7, 1, p2017.sondages.length, k, 0.1)}
							/>
						);
					}),
				)}
			</svg>
			{CANDS.map((cand) => {
				if (t < entrees[cand]) return null;
				const [b, h] = BANDES[cand];
				const a = avance(t, entrees[cand], 0.7);
				const milieu = (b + h) / 2;
				return (
					<Tete
						key={cand}
						nom={cand}
						x={lerp(-8.2, -6.2, a)}
						y={lerp(milieu + 0.4, milieu, a) + 0.35 * Math.sin(Math.PI * a)}
						hauteur={1.3}
						angle={INCLINAISONS[cand] + 0.07 * Math.sin(a * 3 * 2 * Math.PI)}
						opacite={sortie}
					/>
				);
			})}
			<Txt x={960} y={1080 - 0.45 * U} ax={0.5} ay={1} taille={28} police={SERIF} couleur={C['text-primary']} opacite={avance(t, S + 4.6, 0.6) * sortie}>
				{`tous du même côté du résultat${NBSP}: Macron sous-estimé, Le Pen surestimée`}
			</Txt>
		</div>
	);
};

const Titre: React.FC<{t: number}> = ({t}) => {
	const q = avance(t, T.titre, 1.2);
	const {elections, pays, sondages} = donnees.titre;
	return (
		<div style={{position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 0.45 * U}}>
			<div style={{fontFamily: SERIF, fontSize: F(60), lineHeight: 1, color: C['text-primary'], opacity: q, translate: `0 ${0.15 * U * (1 - q)}px`}}>
				Que valent <span style={{fontStyle: 'italic', color: C.accent}}>vraiment</span> les sondages{NBSP}?
			</div>
			<Txt x={0} y={0} taille={17} couleur={C['text-secondary']} opacite={avance(t, T.titre + 1.2, 0.8)} style={{position: 'relative'}}>
				{`${fr(elections, 0)} élections · ${fr(pays, 0)} pays · ${fr(sondages, 0)} sondages confrontés aux résultats`}
			</Txt>
		</div>
	);
};

export const Fin: React.FC = () => {
	const t = useCurrentFrame() / FPS;
	if (t < T.accident) return null;
	if (t < T.p2017) return <Accident t={t} />;
	if (t < T.titre) return <Presidentielle t={t} />;
	return <Titre t={t} />;
};
