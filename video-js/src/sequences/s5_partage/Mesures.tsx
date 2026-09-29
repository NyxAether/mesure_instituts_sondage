// Élection par élection : le consensus d'erreur, puis le resserrement (mimétisme), un point par élection.
import React from 'react';
import {C, G, SERIF} from '../../lib/charte';
import {F, NBSP, Txt, UNITE as U, X, Y, avance, echelonne, fr, largeurMono, lerp} from '../../lib/outils';
import {Boite, Calque, Ligne} from './commun';
import {D, ELECTIONS, MIM, essaim} from './donnees';
import {CONSENSUS, MIMETISME} from './temps';

const RAYON = 0.06;
const G_AX = -5.6;
const D_AX = 3.0;
const Y_C = -0.35;
const Y_R = -2.95;
const R_MIN = 0.1;
const R_MAX = 100;

const xC = (v: number) => G_AX + v * (D_AX - G_AX);
const xR = (v: number) => G_AX + ((Math.log10(v) - Math.log10(R_MIN)) / (Math.log10(R_MAX) - Math.log10(R_MIN))) * (D_AX - G_AX);

/** Un point par élection, empilés en essaim au-dessus de l'axe. */
const positions = (valeurs: number[], yAxe: number, xDe: (v: number) => number) => {
	const xs = valeurs.map(xDe);
	const ys = essaim(xs, 2 * RAYON * 1.1);
	return xs.map((x, i): [number, number] => [x, yAxe + RAYON * 1.3 + ys[i]]);
};

const melange = (a: string, b: string, u: number) => `color-mix(in srgb, ${a} ${Math.round(u * 100)}%, ${b})`;

const POS_C = positions(ELECTIONS.map((e) => e.consensus), Y_C, xC);
const POS_R = positions(ELECTIONS.map((e) => e.resserrement), Y_R, xR);
const ANORMAL_C = ELECTIONS.map((e) => e.rang_consensus < D.seuil);
const ANORMAL_R = ELECTIONS.map((e) => e.rang_resserrement < D.seuil);
const HASARD_C = ELECTIONS.map((e) => e.consensus_hasard);

// Le compteur affiché doit être celui des points colorés.
const part = (drapeaux: boolean[]) => drapeaux.filter(Boolean).length / drapeaux.length;
if (Math.abs(part(ANORMAL_C) - MIM.part_consensus) > 0.005 || Math.abs(part(ANORMAL_R) - MIM.part_resserrement) > 0.005) {
	throw new Error('les parts anormales ne correspondent plus aux points');
}

const Compteur: React.FC<{nom: string; part: number; couleur: string; yAxe: number; opacite: number}> = ({nom, part: p, couleur, yAxe, opacite}) => (
	<Boite x={X(6.6)} y={Y(yAxe + 0.2)} ax={1} ay={1} align="flex-end" ecart={0.08 * U} opacite={opacite}>
		<Ligne taille={15} couleur={C['text-secondary']}>{nom}</Ligne>
		<Ligne taille={56} police={SERIF} couleur={couleur} hauteur={0.75}>
			<span style={{fontStyle: 'italic', position: 'relative', top: 0.08 * U}}>{`${fr(100 * p, 0)}${NBSP}%`}</span>
		</Ligne>
		<Ligne taille={15} couleur={C['text-muted']}>{`attendu au hasard : ${fr(100 * D.seuil, 0)}${NBSP}%`}</Ligne>
	</Boite>
);

export const Mesures: React.FC<{t: number}> = ({t}) => {
	const sortie = 1 - avance(t, MIMETISME.sortie, 0.8);
	const n = ELECTIONS.length;
	const axeC = avance(t, CONSENSUS.axe, 1);
	const axeR = avance(t, MIMETISME.axe, 1);
	const teinteC = avance(t, CONSENSUS.anormaux, 1.5);
	const teinteR = avance(t, MIMETISME.anormaux, 1.5);
	const anneau = avance(t, CONSENSUS.anneau, 0.6) * (1 - avance(t, CONSENSUS.anneauSortie, 0.5));
	const ex = POS_C[D.exemple.index];
	const hasardMin = Math.min(...HASARD_C);
	const hasardMax = Math.max(...HASARD_C);
	const bande = avance(t, CONSENSUS.bande, 0.8);
	const hLab = 0.14 * U; // hauteur d'un chiffre mono de taille 14
	const yBornesC = Y(Y_C) + 0.12 * U + hLab + 0.08 * U;
	const yBornesR = Y(Y_R) + 0.12 * U + hLab + 0.08 * U;
	const demi = (texte: string, taille: number) => largeurMono(texte, taille) / 2;

	return (
		<Calque opacite={sortie}>
			<Calque opacite={axeC}>
				<Txt x={0.55 * U} y={Y(2.07)} ay={0.5} taille={14} couleur={C['text-muted']}>
					{`${fr(MIM.nb_elections, 0)} élections depuis 2000 · sondages des ${MIM.jours_max} derniers jours, au moins ${MIM.sondages_min} par élection · un point par élection`}
				</Txt>
				<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
					<line x1={X(G_AX)} x2={X(D_AX)} y1={Y(Y_C)} y2={Y(Y_C)} stroke={G.axis} strokeWidth={1.5} />
				</svg>
				{[0, 0.5, 1].map((v) => (
					<Txt key={v} x={X(xC(v))} y={Y(Y_C) + 0.12 * U} ax={0.5} taille={14} couleur={C['text-secondary']}>{fr(v, v === 0.5 ? 1 : 0)}</Txt>
				))}
				<Txt x={X(G_AX) - demi('0', 14)} y={yBornesC} taille={13} couleur={C['text-muted']}>erreurs des deux côtés</Txt>
				<Txt x={X(D_AX) + demi('1', 14)} y={yBornesC} ax={1} taille={13} couleur={C['text-muted']}>toutes du même côté</Txt>
				<Txt x={X(G_AX)} y={Y(Y_C + 1.75)} ay={1} taille={15} couleur={C['text-primary']}>
					consensus d’erreur : les sondages se trompent-ils dans le même sens ?
				</Txt>
			</Calque>

			<Calque opacite={bande}>
				<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
					<rect x={X(xC(hasardMin))} y={Y(Y_C + 0.9)} width={X(xC(hasardMax)) - X(xC(hasardMin))} height={0.9 * U} fill={C['text-secondary']} fillOpacity={0.15} />
				</svg>
				<Txt x={X(xC(hasardMin))} y={Y(Y_C + 0.9) - 0.08 * U} ay={1} taille={13} couleur={C['text-secondary']}>
					{`au hasard · autour de ${fr(MIM.consensus_hasard_median, 2)}`}
				</Txt>
			</Calque>

			<Txt x={X(G_AX)} y={Y(Y_C + 1.75) + 0.12 * U} taille={14} couleur={C['text-secondary']} opacite={avance(t, CONSENSUS.points, 2)}>
				{`les vrais sondages · médiane ${fr(MIM.consensus_median, 2)}`}
			</Txt>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				{POS_C.map(([x, y], i) => (
					<circle
						key={i}
						cx={X(x)}
						cy={Y(y)}
						r={RAYON * U}
						fill={ANORMAL_C[i] ? melange(G.series[0], C['text-muted'], teinteC) : C['text-muted']}
						opacity={echelonne(t, CONSENSUS.points, 2, n, i, 0.01)}
					/>
				))}
			</svg>
			<Compteur nom="consensus anormal" part={MIM.part_consensus} couleur={G.series[0]} yAxe={Y_C} opacite={teinteC} />

			<Calque opacite={anneau}>
				<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
					<circle cx={X(ex[0])} cy={Y(ex[1])} r={RAYON * 2.2 * U} fill="none" stroke={C['text-primary']} strokeWidth={2} />
					<line x1={X(ex[0])} x2={X(ex[0])} y1={Y(Y_C + 1.05) + 0.05 * U + F(13) * 0.5 + 0.02 * U} y2={Y(ex[1]) - RAYON * 2.2 * U} stroke={C['text-primary']} strokeWidth={1.5} />
				</svg>
				<Txt x={X(ex[0])} y={Y(Y_C + 1.05)} ax={0.5} ay={0.5} taille={13} couleur={C['text-primary']}>
					{`royaume-uni ${D.exemple.annee}`}
				</Txt>
			</Calque>

			<Calque opacite={axeR}>
				<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
					<line x1={X(G_AX)} x2={X(D_AX)} y1={Y(Y_R)} y2={Y(Y_R)} stroke={G.axis} strokeWidth={1.5} />
					<line x1={X(xR(1))} x2={X(xR(1))} y1={Y(Y_R)} y2={Y(Y_R + 1.1)} stroke={C['text-muted']} strokeWidth={1.5} strokeDasharray="8 8" />
				</svg>
				{[0.1, 1, 10, 100].map((v) => (
					<Txt key={v} x={X(xR(v))} y={Y(Y_R) + 0.12 * U} ax={0.5} taille={14} couleur={C['text-secondary']}>{fr(v, v < 1 ? 1 : 0)}</Txt>
				))}
				<Txt x={X(G_AX) - demi('0,1', 14)} y={yBornesR} taille={13} couleur={C['text-muted']}>plus semblables</Txt>
				<Txt x={X(xR(1))} y={yBornesR} ax={0.5} taille={13} couleur={C['text-muted']}>= hasard</Txt>
				<Txt x={X(D_AX) + demi('100', 14)} y={yBornesR} ax={1} taille={13} couleur={C['text-muted']}>plus dispersés</Txt>
				<Txt x={X(G_AX)} y={Y(Y_R + 1.25)} ay={1} taille={15} couleur={C['text-primary']}>
					resserrement : les sondages se ressemblent-ils trop ? (mimétisme)
				</Txt>
			</Calque>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				{POS_R.map(([x, y], i) => (
					<circle
						key={i}
						cx={X(x)}
						cy={Y(y)}
						r={RAYON * U * (ANORMAL_R[i] ? lerp(1, 1.3, teinteR) : 1)}
						fill={ANORMAL_R[i] ? melange(G.series[1], C['text-muted'], teinteR) : C['text-muted']}
						opacity={echelonne(t, MIMETISME.points, 1.5, n, i, 0.01)}
					/>
				))}
			</svg>
			<Compteur nom="resserrement anormal" part={MIM.part_resserrement} couleur={G.series[1]} yAxe={Y_R} opacite={teinteR} />
		</Calque>
	);
};
